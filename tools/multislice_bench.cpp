// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
//
// Microbenchmark for the dynamic-lookup rendering experiment (branch
// jdl-experiment). Measures the raw cost of a popops::multiSlice gather sized
// like one frame of 3DGS routing, to decide whether a gather-based renderer
// could replace NEWS routing. See jdl_multislice_experiment.md.
//
// It is fully self-contained (depends only on poplar + popops), so it cannot
// affect the renderer. Build the `multislice_bench` target in the container:
//     ninja multislice_bench
//     ./multislice_bench [numGaussians] [perTile]
//
// NOTE: the popops::embedding / multiSlice planning API has shifted slightly
// across Poplar releases. If it does not compile against SDK 3.3, the spots to
// check are flagged with `[API]` below.

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include <popops/codelets.hpp>
#include <popops/DynamicSlice.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace poplar;

int main(int argc, char** argv) {
  // Workload parameters. Defaults model a bonsai-sized scene routed over the
  // 1440-tile framebuffer with the current NEWS channel capacity (400/tile).
  unsigned numGaussians  = (argc > 1) ? std::stoul(argv[1]) : 44000u; // table entries
  unsigned perTile       = (argc > 2) ? std::stoul(argv[2]) : 400u;   // lookups per tile
  const unsigned numTiles      = 1440u;
  const unsigned gaussianFloats = 15u;            // sizeof(Gaussian3D)/4 == 60B/4
  const unsigned numLookups    = numTiles * perTile;
  const unsigned reps          = 200u;            // amortise the engine.run barrier

  std::cout << "multiSlice gather benchmark\n"
            << "  table:   " << numGaussians << " entries x " << gaussianFloats << " floats\n"
            << "  lookups: " << numLookups   << " (" << numTiles << " tiles x " << perTile << ")\n"
            << "  reps:    " << reps << "\n";

  // -- attach an IPU --
  auto dm = DeviceManager();
  auto devs = dm.getDevices(TargetType::IPU, 1);
  Device device;
  bool attached = false;
  for (auto& d : devs) {
    if (d.attach()) { device = std::move(d); attached = true; break; }
  }
  if (!attached) { std::cerr << "Could not attach to an IPU.\n"; return 1; }
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  if (numTiles > target.getNumTiles()) {
    std::cerr << "numTiles " << numTiles << " > target tiles "
              << target.getNumTiles() << "\n";
    return 1;
  }

  // -- plan + build the gather -- [API]
  // plan(graph, dataType, numEntries, outputSize, numLookupsPerStep, options)
  auto plan = popops::embedding::plan(graph, FLOAT, numGaussians, gaussianFloats,
                                      {numLookups}, {});

  // Sliceable table laid out as the plan prefers, and a matching indices tensor.
  Tensor table = popops::createSliceableTensor(
      graph, FLOAT, {numGaussians, gaussianFloats}, {0}, {1}, plan, {}, "gaussian_table");
  Tensor offsets = popops::createIndicesTensor(graph, {0}, numLookups, plan, {}, "offsets");
  graph.createHostWrite("offsets_h", offsets);

  // ---- gather (transport only) ----
  program::Sequence gather;
  Tensor result = popops::multiSlice(graph, table, offsets, {0}, {1}, gather, plan, {},
                                     "gather");

  // ---- output mapping: pin each tile's perTile rows onto that tile ----
  // This is the layout the blend step needs: tile t owns rows [t*perTile,
  // (t+1)*perTile). The offsets are ordered so result row (t*perTile + k) is
  // tile t's k-th Gaussian, so reshape + Copy delivers each tile's set to it.
  Tensor pinned = graph.addVariable(FLOAT, {numTiles, perTile, gaussianFloats}, "pinned");
  for (unsigned t = 0; t < numTiles; ++t) graph.setTileMapping(pinned[t], t);

  program::Sequence gatherAndPin;
  Tensor result2 = popops::multiSlice(graph, table, offsets, {0}, {1}, gatherAndPin, plan, {},
                                      "gather2");
  // result2 is [numLookups, 1, gaussianFloats]; reshape to [numTiles, perTile, gf]
  // (metadata-only, preserves flat order) then Copy into the per-tile layout.
  gatherAndPin.add(program::Copy(
      result2.reshape({numTiles, perTile, gaussianFloats}), pinned));

  // How spread is each layout? Count distinct tiles holding rows.
  auto countTiles = [&](const Tensor& tn) {
    auto m = graph.getTileMapping(tn);
    unsigned n = 0;
    for (const auto& iv : m) if (!iv.empty()) ++n;
    return n;
  };
  std::cout << "  result spread:  " << countTiles(result) << " tiles (plan-chosen)\n";
  std::cout << "  pinned spread:  " << countTiles(pinned) << " tiles (1 per render tile)\n";

  program::Sequence mainGather;
  mainGather.add(program::Repeat(reps, gather));
  program::Sequence mainPin;
  mainPin.add(program::Repeat(reps, gatherAndPin));

  // Raw-poplar Engine: a vector of programs, run by index (0 = gather, 1 = pin).
  std::vector<program::Program> progs = {mainGather, mainPin};
  Engine engine(graph, progs);
  engine.load(device);

  // Random lookup indices (reused every rep — fine for timing the exchange).
  std::mt19937 rng(0);
  std::uniform_int_distribution<unsigned> dist(0, numGaussians - 1);
  std::vector<unsigned> offs(numLookups);
  for (auto& o : offs) o = dist(rng);
  engine.writeTensor("offsets_h", offs.data(), offs.data() + offs.size());

  auto timeProg = [&](unsigned id) {
    auto t0 = std::chrono::steady_clock::now();
    engine.run(id);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  };

  double gatherMs = timeProg(0);
  double pinMs    = timeProg(1);
  double bytes = double(numLookups) * gaussianFloats * sizeof(float);
  double mib = bytes / (1024.0 * 1024.0);

  std::cout << "\n  gather only:      " << gatherMs << " ms/frame ("
            << (mib / (gatherMs / 1000.0)) << " MiB/s)\n";
  std::cout << "  gather + pin:     " << pinMs << " ms/frame\n";
  std::cout << "  pin (rearrange):  " << (pinMs - gatherMs) << " ms/frame\n";
  std::cout << "Compare gather+pin ms against the NEWS `single_exchange` time "
               "from `--benchmark`.\n";
  return 0;
}
