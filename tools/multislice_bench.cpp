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

  // -- plan + build the gather -- [API]
  // plan(graph, dataType, numEntries, outputSize, numLookupsPerStep, options)
  auto plan = popops::embedding::plan(graph, FLOAT, numGaussians, gaussianFloats,
                                      {numLookups}, {});

  // Sliceable table laid out as the plan prefers, and a matching indices tensor.
  Tensor table = popops::createSliceableTensor(
      graph, FLOAT, {numGaussians, gaussianFloats}, {0}, {1}, plan, {}, "gaussian_table");
  Tensor offsets = popops::createIndicesTensor(graph, {0}, numLookups, plan, {}, "offsets");
  graph.createHostWrite("offsets_h", offsets);

  program::Sequence gather;
  Tensor result = popops::multiSlice(graph, table, offsets, {0}, {1}, gather, plan, {},
                                     "gather");
  std::cerr << "result shape: ";
  for (auto s : result.shape()) std::cerr << s << " ";
  std::cerr << "\n";

  program::Sequence main;
  main.add(program::Repeat(reps, gather));

  Engine engine(graph, main);
  engine.load(device);

  // Random lookup indices (reused every rep — fine for timing the exchange).
  std::mt19937 rng(0);
  std::uniform_int_distribution<unsigned> dist(0, numGaussians - 1);
  std::vector<unsigned> offs(numLookups);
  for (auto& o : offs) o = dist(rng);
  engine.writeTensor("offsets_h", offs.data(), offs.data() + offs.size());

  auto t0 = std::chrono::steady_clock::now();
  engine.run(0);
  auto t1 = std::chrono::steady_clock::now();

  double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double perGatherMs = totalMs / reps;
  double bytes = double(numLookups) * gaussianFloats * sizeof(float);

  std::cout << "  total wall: " << totalMs << " ms over " << reps << " reps\n";
  std::cout << "  per gather: " << perGatherMs << " ms\n";
  std::cout << "  gathered:   " << (bytes / (1024.0 * 1024.0)) << " MiB/gather, "
            << (bytes / (1024.0 * 1024.0) / (perGatherMs / 1000.0)) << " MiB/s\n";
  std::cout << "Compare per-gather ms against the NEWS `single_exchange` time "
               "from `--benchmark`.\n";
  return 0;
}
