// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/ipu_rasteriser.hpp>
#include <splat/geometry.hpp>
#include <ipu/io_utils.hpp>
#include <cstring>
#include <opencv2/highgui.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <poputil/TileMapping.hpp>

#include <popops/codelets.hpp>
#include <popops/TopK.hpp>
#include <popops/Fill.hpp>
#include <popops/DynamicSlice.hpp>

#include <tileMapping/edge_builder.hpp>
#include <splat/gather_discovery.hpp>

using namespace poplar;

namespace splat {

IpuSplatter::IpuSplatter(const Points& verts, TiledFramebuffer& fb, bool noAMP)
  : modelView("mv"), projection("mp"), fxy("fxy"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"),
    counts("splat_counts"), continueFlag("continue_flag"),
    phaseCyclesStream("phase_cycles"),
    hostModelView(16),
    hostProjection(16),
    fxyHost(2),
    hostContinueFlag(1, 1),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  hostVertices.reserve(4 * verts.size());
  printf("VERTS size: %luB\n", verts.size());
  for (const auto& v : verts) {
    hostVertices.push_back(v.p.x);
    hostVertices.push_back(v.p.y);
    hostVertices.push_back(v.p.z);
    hostVertices.push_back(1.f);
  }
  frameBuffer.resize(fb.width * fb.height * 4, 0); // RGBX: 4 bytes per pixel for 32-bit alignment
  printf("Fb size: %luB\n", frameBuffer.size());

  splatCounts.resize(fb.numTiles, 0);
  phaseCycleData.resize(fb.numTiles * 5, 0);
}

IpuSplatter::IpuSplatter(const Gaussians& verts, TiledFramebuffer& fb, bool noAMP)
  : modelView("mv"), projection("mp"), fxy("fxy"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"),
    counts("splat_counts"), continueFlag("continue_flag"),
    phaseCyclesStream("phase_cycles"),
    hostModelView(16),
    hostProjection(16),
    fxyHost(2),
    hostContinueFlag(1, 1),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  gaussiansHost = verts;  // kept for per-frame host discovery in gather mode

  auto elemSize = sizeof(verts[0]);
  hostVertices.reserve(elemSize * verts.size());
  printf("num verts in: %lu, elemsize: %lu \n", verts.size(), elemSize);

  for (auto j = 0u; j < verts.size(); ++j) {
    auto gptr = (const float*)&verts[j];
    for (auto i = 0u; i < elemSize; ++i) {
      hostVertices.push_back(*(gptr + i));
    }
  }

  frameBuffer.resize(fb.width * fb.height * 4, 0); // RGBX: 4 bytes per pixel for 32-bit alignment
  printf("Fb size: %luB\n", frameBuffer.size());

  splatCounts.resize(fb.numTiles);
  for (auto& c : splatCounts) {
    c = 0;
  }
  phaseCycleData.resize(fb.numTiles * 5, 0);
}


void IpuSplatter::updateModelView(const glm::mat4& mv) {
  currentView = mv;  // codelet reconstructs exactly this; host discovery needs it
  auto mvt = glm::transpose(mv);
  auto ptr = (const float*)glm::value_ptr(mvt);
  for (auto i = 0u; i < hostModelView.size(); ++i) {
    hostModelView[i] = *ptr;
    ptr += 1;
  }
}

void IpuSplatter::updateProjection(const glm::mat4& mp) {
  currentProj = mp;
  auto mpt = glm::transpose(mp);
  auto ptr = (const float*)glm::value_ptr(mpt);
  for (auto i = 0u; i < hostProjection.size(); ++i) {
    hostProjection[i] = *ptr;
    ptr += 1;
  }
}

void IpuSplatter::getIPUHistogram(std::vector<u_int32_t>& counts) const {
  counts = splatCounts;
}

void IpuSplatter::updateFocalLengths(float fx, float fy) {
  currentFov = fx;  // half-FOV in radians (fxy[0] in the codelet)
  fxyHost = {fx, fy};
}

// takes a cv::Mat image and returns a copy of the original but with the image partitioned into tiles of size tileHeight x tileWidth.
// dataType is the data type of the image (e.g. CV_8UC3 for 8-bit unsigned char 3-channel image)
// It treats the image as one vector of pixels which we pick from in chunks of tileHeight x tileWidth
cv::Mat tileImageBuffer(cv::Mat image, int tileHeight, int tileWidth, int dataType, int channels) {
    cv::Mat new_image(image.rows, image.cols, dataType);
    uchar *buffer = image.data;
    int stripSize = tileHeight * tileWidth;
     
    for (int j = 0; j < int(floor(image.rows / float(tileHeight))); j++) {
        for (int i = 0; i < int(floor(image.cols / float(tileWidth))); i++) {
            cv::Mat chunk(tileHeight, tileWidth, dataType, buffer, cv::Mat::AUTO_STEP);
            chunk.copyTo(new_image(cv::Rect(i * tileWidth, j * tileHeight, tileWidth, tileHeight)));
            buffer += stripSize * channels;
        }
    }

    return new_image;
}

void IpuSplatter::getFrameBuffer(cv::Mat &frame) const {
  // Framebuffer is uint8 RGBX (4 bytes/pixel, X=padding) from the IPU.
  // Rearrange tile strips, then convert RGBA->BGR (the X channel is ignored by OpenCV).
  cv::Mat image_rgba = cv::Mat(cv::Size(fbMapping.width, fbMapping.height), CV_8UC4, (void *) frameBuffer.data(), cv::Mat::AUTO_STEP);
  cv::Mat tiled = tileImageBuffer(image_rgba, fbMapping.tileHeight, fbMapping.tileWidth, CV_8UC4, 4);
  cvtColor(tiled, frame, cv::COLOR_RGBA2BGR);
}

void IpuSplatter::getProjectedPoints(std::vector<glm::vec4>& pts) const {
  pts.resize(hostVertices.size() / 4);
  const auto* ptr = hostVertices.data();
  for (auto i = 0u; i < pts.size(); ++i) {
    pts[i].x = *(ptr + 0);
    pts[i].y = *(ptr + 1);
    pts[i].z = *(ptr + 2);
    pts[i].w = *(ptr + 3);
    ptr += 4;
  }
}

struct MappingInfo {
  std::size_t padding;
  std::size_t elementsPerTile;
  std::size_t totalTiles;
};

MappingInfo calculateMapping(poplar::Graph& g, std::size_t numElements, std::size_t grainSize, TiledFramebuffer &fbMapping) {
  ipu_utils::logger()->info("Input size of data: {}B", numElements);
  const double numTiles = g.getTarget().getNumTiles();

  if (fbMapping.numTiles < numTiles) {
    ipu_utils::logger()->info("Number of tiles in framebuffer ({}) is less than number of tiles on target ({})", fbMapping.numTiles, numTiles);
  }

  double grainsPerTile = std::ceil(numElements / (fbMapping.numTiles * grainSize));
  double elementsPerTile = grainsPerTile * grainSize;
  double fullTiles = std::floor(numElements / elementsPerTile);
  double unfilledTiles = fbMapping.numTiles - fullTiles;
  double remainingElements = numElements - (fullTiles * elementsPerTile);
  double paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;

  auto totalTiles = fullTiles + unfilledTiles;
  totalTiles = totalTiles > fbMapping.numTiles - 1 ? fbMapping.numTiles - 1 : totalTiles;

  ipu_utils::logger()->info("Upper bound elements per tile: {}", elementsPerTile);
  ipu_utils::logger()->info("Full tiles: {}", fullTiles);
  ipu_utils::logger()->info("Unfilled tiles: {}", unfilledTiles);
  ipu_utils::logger()->info("Remaining elements: {}", remainingElements);
  ipu_utils::logger()->info("Padded elements on last used tile: {}", paddedRemainder);
  ipu_utils::logger()->info("Padding: {}", paddedRemainder - remainingElements);
  ipu_utils::logger()->info("Total padding to fill all tiles: {}", paddedRemainder - remainingElements + (unfilledTiles * elementsPerTile));
  ipu_utils::logger()->info("Total tiles: {}", totalTiles);

  const std::size_t padding = paddedRemainder - remainingElements + (unfilledTiles * elementsPerTile);
  return MappingInfo{padding, std::size_t(elementsPerTile), std::size_t(totalTiles)};
}


/// Distribute elements across tiles such that the number of elements on a
/// tile is always divisible by grainSize. The tensor must have already been padded
/// to a multiple of grain size.
void applyTileMapping(poplar::Graph& g, const poplar::Tensor& paddedInput, const MappingInfo& info) {
  auto sliceStart = 0u;
  auto t = 0u;
  for (t; t < info.totalTiles; ++t) {
    const auto sliceEnd = sliceStart + info.elementsPerTile;
    g.setTileMapping(paddedInput.slice(sliceStart, sliceEnd), t);
    sliceStart = sliceEnd;
  }

  // Last tile has fewer elements:
  auto lastSlice = paddedInput.slice(sliceStart, paddedInput.numElements());
  ipu_utils::logger()->info("Size of slice on last tile: {}", lastSlice.numElements());
  if (lastSlice.numElements() > 0) {
    g.setTileMapping(lastSlice, t);
  }
}


void IpuSplatter::build(poplar::Graph& graph, const poplar::Target& target) {
  if (gatherMode) { buildGatherPath(graph, target); return; }

  auto vg = graph.createVirtualGraph(0u, fbMapping.numTiles);

  const auto codeletFile = std::string(POPC_PREFIX) + "/codelets/splat/codelets.cpp";
  const auto glmPath = std::string(POPC_PREFIX) + "/external/glm/";
  const auto mathPath = std::string(POPC_PREFIX) + "/include/math";
  const auto otherIncludes = std::string(POPC_PREFIX) + "/include/missing";
  const auto tileMapping = std::string(POPC_PREFIX) + "/include/tileMapping";
  const auto includes = " -I " + glmPath + " -I " + mathPath + " -I " + otherIncludes + " -I " + tileMapping;
  ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
  popops::addCodelets(vg);
  vg.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3 -finline-functions -funroll-loops" + includes);

  // Create storage for the model view projeciton matrix. Place the master copy on tile 0
  // and then broadcast from their to all other tiles before any computations.
  modelView.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(modelView, 0u);

  projection.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(projection, 0u);

  fxy.buildTensor(vg, FLOAT, {2});
  vg.setTileMapping(fxy, 0u);

  // Build a program to upload and broadcast the modelling-projection matrix:
  program::Sequence broadcastMvp;
  broadcastMvp.add(modelView.buildWrite(vg, true));
  broadcastMvp.add(projection.buildWrite(vg, true));
  broadcastMvp.add(fxy.buildWrite(vg, true));

  auto fbGrainSize = 4; // 4 unsigned chars per pixel (RGBX, 32-bit aligned)
  auto fbToTileMapping = calculateMapping(vg, frameBuffer.size(), fbGrainSize, fbMapping);
  ipu_utils::logger()->info("Framebuffer layout: padding: {}, elementsPerTile: {}, totalTiles: {}", fbToTileMapping.padding, fbToTileMapping.elementsPerTile, fbToTileMapping.totalTiles);
  auto paddedFramebuffer = vg.addVariable(UNSIGNED_CHAR, {frameBuffer.size() + fbToTileMapping.padding}, "padded_frame_buffer");
  applyTileMapping(vg, paddedFramebuffer, fbToTileMapping);
  outputFramebuffer = paddedFramebuffer.slice(0, frameBuffer.size());

  // Map the point cloud vertices across all tiles. TODO: If we are not using AMP the only constraint
  // is that the grain size must be a multiple of 4 (so that 4-vectors are not split between
  // tiles). If we use the AMP we need to have at least 8 4-vectors to fill the AMP pipeline so
  // the minimum grain size is 32:
  ipu_utils::logger()->info("hostvertices size: {}, numtiles {}", hostVertices.size(), fbToTileMapping.totalTiles);
  const auto grainSize = sizeof(Gaussian3D); //disableAMPVertices ? 4 : 4 * 8;
  auto mapping = calculateMapping(vg, hostVertices.size(), grainSize, fbMapping);
  ipu_utils::logger()->info("Vertex layout: padding: {}, elementsPerTile: {}, totalTiles: {}", mapping.padding, mapping.elementsPerTile, mapping.totalTiles);
  auto paddedInput = vg.addVariable(FLOAT, {hostVertices.size() + mapping.padding}, "padded_verts_in");
  applyTileMapping(vg, paddedInput, mapping);
  // We only want to stream to a slice of the padded tensor:
  inputVertices = paddedInput.slice(0, hostVertices.size());

  // Two compute sets: route (single-worker projection/routing) then blend (6-worker pixel blending).
  // Poplar's BSP barrier between compute sets replaces the software barrier.
  auto routeCs = vg.addComputeSet("route");
  auto blendCs = vg.addComputeSet("blend");

  // Per-tile capacity tuning. NOTE: the storage tensors are declared as
  //   addVariable(poplar::FLOAT, {extraStorageSize})  // length in FLOATS
  // so `extraStorageSize` is in float elements, NOT bytes. Each Gaussian
  // occupies sizeof(Gaussian3D) FLOATS in the buffer (== 60 floats == 240
  // bytes), of which only 15 floats (60 B) are real data — the loader pushes
  // sizeof(struct) floats per Gaussian, the codelet steps by the same
  // amount. So per +1 to numPoints (with multiplier M):
  //   8 channels   :   8 * 60 B               = 480 B
  //   extra_storage:   60 * M floats * 4      = 240*M B
  //   z-buffer     :   40 * M floats * 4      = 160*M B
  //   indices      :    4 * M B               =   4*M B
  // = 480 + 404*M  bytes per tile per +1 numPoints.
  //
  // The original numPoints=360 / multiplier=2 already used ~470 KB of the
  // 624 KB tile (plus ~130 KB code + stacks), leaving little headroom. A
  // small bump is the realistic ceiling without a deeper packing fix.
  // numPoints=400 / multiplier=2 uses ~50 KB more — usually fits; if popc
  // complains, drop to 380.
  unsigned numPoints = 400;
  std::size_t channelSize = numPoints * grainSize;
  std::size_t extraStorageSize = channelSize * 2;

  // construct z-buffer program to sort the gaussians
  program::Sequence sortGaussians;

  MappingInfo zBufferMapping = mapping;
  zBufferMapping.elementsPerTile = (zBufferMapping.elementsPerTile + extraStorageSize) / grainSize;
  std::size_t totalGaussianCapacity = (hostVertices.size() + mapping.padding + extraStorageSize * zBufferMapping.totalTiles) / grainSize;
  
  const auto indices = vg.addVariable(poplar::INT, {totalGaussianCapacity}, "indices");
  applyTileMapping(vg, indices, zBufferMapping);

  const auto splatCounts = vg.addVariable(poplar::UNSIGNED_INT, {(size_t) fbMapping.numTiles});
  MappingInfo counterInfo = {0, 1, (size_t) fbMapping.numTiles};
  applyTileMapping(vg, splatCounts, counterInfo);
  counts = splatCounts.slice(0, fbMapping.numTiles);

  const size_t NUM_PHASES = 5;
  auto phaseCycleTensor = vg.addVariable(poplar::UNSIGNED_INT, {(size_t)fbMapping.numTiles * NUM_PHASES}, "phase_cycles");
  MappingInfo phaseInfo = {0, NUM_PHASES, (size_t)fbMapping.numTiles};
  applyTileMapping(vg, phaseCycleTensor, phaseInfo);
  phaseCyclesStream = phaseCycleTensor;

  std::vector<poplar::VertexRef> vertices;
  // Get the tile mapping and connect the vertices:
  const auto tm = vg.getTileMapping(paddedInput);
  const auto tmFb = vg.getTileMapping(paddedFramebuffer);
  const auto tmIndices = vg.getTileMapping(indices);
  const auto tmCounts = vg.getTileMapping(splatCounts);

  for (auto t = 0u; t < tm.size(); ++t) {
    const auto& m = tm[t];
    const auto& mFb = tmFb[t];
    const auto& mIndices = tmIndices[t];
    const auto& mCounts = tmCounts[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected fb to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      // Add the tile local MVP matrix variable and append a copies that broadcast it to all tiles:
      auto localMv = vg.clone(modelView, "mv_tile_" + std::to_string(t));
      vg.setTileMapping(localMv, t);
      broadcastMvp.add(program::Copy(modelView, localMv));

      auto localProj = vg.clone(projection, "mp_tile_" + std::to_string(t));
      vg.setTileMapping(localProj, t);
      broadcastMvp.add(program::Copy(projection, localProj));

      auto localFxy = vg.clone(fxy, "fxy_tile_" + std::to_string(t));
      vg.setTileMapping(localFxy, t);
      broadcastMvp.add(program::Copy(fxy, localFxy));

      auto ptsIn = paddedInput.slice(m.front());
      auto sliceFb = paddedFramebuffer.slice(mFb.front());
      auto sliceIdxs = indices.slice(mIndices.front());
      auto counter = splatCounts.slice(mCounts.front());

      auto storage = vg.addVariable(poplar::FLOAT, {extraStorageSize}, "extra_storage");
      vg.setTileMapping(storage, t);
      auto gaussians = concat(ptsIn, storage);

      auto gaus2D = vg.addVariable(poplar::FLOAT, {sliceIdxs.numElements() * sizeof(Gaussian2D)}, "z_buffer");
      vg.setTileMapping(gaus2D, t);

      auto tid = vg.addConstant<int>(INT, {1}, {int(t)});
      vg.setTileMapping(tid, t);


      auto rv = vg.addVertex(routeCs, "RouteVertex");
      vg.setTileMapping(rv, t);
      vg.connect(rv["modelView"], localMv.flatten());
      vg.connect(rv["projection"], localProj.flatten());
      vg.connect(rv["vertsIn"], gaussians);
      vg.connect(rv["indices"], sliceIdxs);
      vg.connect(rv["gaus2D"], gaus2D);
      vg.connect(rv["fxy"], localFxy);
      vg.connect(rv["tile_id"], tid);
      vg.connect(rv["splatted"], counter);
      vg.connect(rv["phaseCycles"], phaseCycleTensor.slice(t * NUM_PHASES, (t + 1) * NUM_PHASES));
      vertices.push_back(rv);

      auto bv = vg.addVertex(blendCs, "BlendVertex");
      vg.setTileMapping(bv, t);
      vg.connect(bv["gaus2D"], gaus2D);
      vg.connect(bv["splatted"], counter);
      vg.connect(bv["tile_id"], tid);
      vg.connect(bv["localFb"], sliceFb);
    }
  }


  EdgeBuilder eb(vg, vertices, channelSize);
  eb.constructLattice(tm, fbMapping);
  // this program sequence will copy the points between all the tiles in the graph
  program::Sequence broadcastPoints = eb.getBroadcastSequence();

  // Each repeat iteration runs one full compute→exchange cycle. Gaussians
  // travel one hop per iteration, so routingRepeats controls how many hops
  // settle per frame. K=1 is the original behaviour. K=2-3 halves/thirds
  // the routing spike after a view change. Compute is ~9ms and streaming
  // is the bottleneck, so moderate K values cost little wall-clock time.
  constexpr unsigned routingRepeats = 1;

  program::Sequence substep;
  substep.add(program::Execute(routeCs));
  substep.add(program::Execute(blendCs));
  substep.add(broadcastPoints);

  auto readFb = outputFramebuffer.buildRead(vg, true);
  auto readCounts = counts.buildRead(vg, true);
  auto readPhaseCycles = phaseCyclesStream.buildRead(vg, true);

  program::Sequence main;
  main.add(broadcastMvp);
  main.add(program::Repeat(routingRepeats, substep));
  main.add(readFb);
  main.add(readCounts);
  main.add(readPhaseCycles);

  // Device-side render loop: runs on-device until host sets flag to 0.
  // Eliminates per-frame host↔device engine.run() barrier.
  auto flagTensor = vg.addVariable(INT, {}, "continue_flag");
  vg.setTileMapping(flagTensor, 0u);
  continueFlag = flagTensor.reshape({1});

  auto condProgram = continueFlag.buildWrite(vg, true);

  program::Sequence frameBody;
  frameBody.add(broadcastMvp);
  frameBody.add(program::Repeat(routingRepeats, substep));
  frameBody.add(readFb);
  frameBody.add(readCounts);
  frameBody.add(readPhaseCycles);

  program::Sequence setup;
  setup.add(inputVertices.buildWrite(vg, true));

  getPrograms().add("write_verts", setup);
  getPrograms().add("project", main);
  getPrograms().add("render_loop",
      program::RepeatWhileTrue(condProgram, flagTensor, frameBody));

  getPrograms().add("broadcast_mvp", broadcastMvp);
  getPrograms().add("single_route", program::Execute(routeCs));
  getPrograms().add("single_blend", program::Execute(blendCs));
  getPrograms().add("single_exchange", broadcastPoints);
  getPrograms().add("read_fb", readFb);
  getPrograms().add("read_counts", readCounts);
  getPrograms().add("read_phase_cycles", readPhaseCycles);
}

void IpuSplatter::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (gatherMode) { executeGather(engine, device); return; }

  if (!initialised) {
    initialised = true;
    enginePtr = &engine;
    modelView.connectWriteStream(engine, hostModelView);
    projection.connectWriteStream(engine, hostProjection);
    fxy.connectWriteStream(engine, fxyHost);
    inputVertices.connectWriteStream(engine, hostVertices);
    outputFramebuffer.connectReadStream(engine, frameBuffer);
    counts.connectReadStream(engine, splatCounts);
    continueFlag.connectWriteStream(engine, hostContinueFlag);
    phaseCyclesStream.connectReadStream(engine, phaseCycleData);
    getPrograms().run(engine, "write_verts");
  }

  if (deviceLoopMode && !deviceLoopRunning.load()) {
    hostContinueFlag[0] = 1;
    deviceLoopRunning = true;
    deviceThread = std::thread([this, &engine]() {
      getPrograms().run(engine, "render_loop");
      deviceLoopRunning = false;
    });
    return;
  }

  if (deviceLoopMode) {
    return;
  }

  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  getPrograms().run(engine, "project");
  auto t1 = clk::now();
  lastTiming.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void IpuSplatter::broadcastMVP() {
  getPrograms().run(*enginePtr, "broadcast_mvp");
}

PhaseTiming IpuSplatter::runSingleSubstep() {
  using clk = std::chrono::steady_clock;
  PhaseTiming t;

  auto t0 = clk::now();
  getPrograms().run(*enginePtr, "single_route");
  auto t1 = clk::now();
  getPrograms().run(*enginePtr, "single_blend");
  auto t2 = clk::now();
  getPrograms().run(*enginePtr, "single_exchange");
  auto t3 = clk::now();

  t.route_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  t.blend_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  t.exchange_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  t.compute_ms = t.route_ms + t.blend_ms + t.exchange_ms;
  return t;
}

void IpuSplatter::readbackCounts() {
  getPrograms().run(*enginePtr, "read_counts");
}

void IpuSplatter::readbackFramebuffer() {
  getPrograms().run(*enginePtr, "read_fb");
}

void IpuSplatter::readbackPhaseCycles() {
  getPrograms().run(*enginePtr, "read_phase_cycles");
}

IpuSplatter::CycleBreakdown IpuSplatter::getPhaseCycleStats() const {
  const double clockGHz = 1.85;
  const double toMs = 1.0 / (clockGHz * 1e6);
  const size_t N = 5;
  const size_t numTiles = phaseCycleData.size() / N;

  double sums[5] = {};
  double mins[5], maxs[5];
  for (size_t p = 0; p < N; ++p) {
    mins[p] = 1e30;
    maxs[p] = 0;
  }

  for (size_t t = 0; t < numTiles; ++t) {
    for (size_t p = 0; p < N; ++p) {
      double v = phaseCycleData[t * N + p] * toMs;
      sums[p] += v;
      if (v < mins[p]) mins[p] = v;
      if (v > maxs[p]) maxs[p] = v;
    }
  }

  CycleBreakdown bd;
  PhaseStats* phases[] = {&bd.clear, &bd.routing, &bd.projection, &bd.sorting, &bd.total};
  for (size_t p = 0; p < N; ++p) {
    phases[p]->min_ms = mins[p];
    phases[p]->mean_ms = sums[p] / numTiles;
    phases[p]->max_ms = maxs[p];
  }
  return bd;
}

unsigned IpuSplatter::getTotalSplatCount() const {
  unsigned total = 0;
  for (auto c : splatCounts) total += c;
  return total;
}

void IpuSplatter::stopDeviceLoop() {
  if (deviceLoopRunning.load()) {
    hostContinueFlag[0] = 0;
    if (deviceThread.joinable()) {
      deviceThread.join();
    }
    deviceLoopRunning = false;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: multiSlice gather render path (branch jdl-experiment).
//
// Discovery is done on the host each frame (computeGatherOffsets): every
// Gaussian is assigned to the tiles its 3-sigma bbox overlaps, producing a
// per-tile list of global indices. The device then:
//   1. multiSlice-gathers each tile's Gaussians from a sliceable table,
//   2. pins them onto the owning tile (Copy into a per-tile tensor),
//   3. projects + sorts them (GatherProjectVertex),
//   4. alpha-blends (BlendVertex, unchanged).
// No NEWS channels, no routing, no multi-substep convergence -> no flicker,
// instant settle. The trade is that discovery runs on the host (the deliberate
// "fully on-chip -> NEWS; this speed-up -> host discovery + gather" choice).
//
// Built on the top-level `graph` (not a virtual graph) so popops' multiSlice
// planning sees the real target; per-tile work is mapped to tiles [0, numTiles).
// This is a first integration cut — expect to compile-iterate on the container.
// ---------------------------------------------------------------------------
void IpuSplatter::buildGatherPath(poplar::Graph& graph, const poplar::Target& target) {
  using namespace poplar;
  const unsigned numTiles    = (unsigned)fbMapping.numTiles;
  const unsigned perTile     = gatherPerTile;
  constexpr unsigned GF      = sizeof(Gaussian3D) / sizeof(float);   // 15
  const unsigned numGaussians = (unsigned)gaussiansHost.size();
  const unsigned numLookups  = numTiles * perTile;

  ipu_utils::logger()->info("Gather path: {} gaussians, {} tiles, {}/tile, {} lookups",
                            numGaussians, numTiles, perTile, numLookups);

  popops::addCodelets(graph);
  const auto codeletFile   = std::string(POPC_PREFIX) + "/codelets/splat/codelets.cpp";
  const auto glmPath       = std::string(POPC_PREFIX) + "/external/glm/";
  const auto mathPath      = std::string(POPC_PREFIX) + "/include/math";
  const auto otherIncludes = std::string(POPC_PREFIX) + "/include/missing";
  const auto tileMapping   = std::string(POPC_PREFIX) + "/include/tileMapping";
  const auto includes = " -I " + glmPath + " -I " + mathPath + " -I " + otherIncludes + " -I " + tileMapping;
  graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3 -finline-functions -funroll-loops" + includes);

  // --- MVP master tensors (host-written each frame) + per-tile broadcast ---
  Tensor mvMaster = graph.addVariable(FLOAT, {16}, "g_mv_master");
  Tensor mpMaster = graph.addVariable(FLOAT, {16}, "g_mp_master");
  Tensor fxyMaster = graph.addVariable(FLOAT, {2}, "g_fxy_master");
  graph.setTileMapping(mvMaster, 0u);
  graph.setTileMapping(mpMaster, 0u);
  graph.setTileMapping(fxyMaster, 0u);
  graph.createHostWrite("g_mv_h", mvMaster);
  graph.createHostWrite("g_mp_h", mpMaster);
  graph.createHostWrite("g_fxy_h", fxyMaster);

  program::Sequence broadcastMvp;

  // --- framebuffer (same tiled layout as the NEWS path) ---
  auto fbToTileMapping = calculateMapping(graph, frameBuffer.size(), 4, fbMapping);
  auto paddedFramebuffer = graph.addVariable(UNSIGNED_CHAR,
      {frameBuffer.size() + fbToTileMapping.padding}, "g_padded_fb");
  applyTileMapping(graph, paddedFramebuffer, fbToTileMapping);
  auto outFb = paddedFramebuffer.slice(0, frameBuffer.size());
  graph.createHostRead("g_fb_h", outFb);
  const auto tmFb = graph.getTileMapping(paddedFramebuffer);

  // --- sliceable Gaussian table + per-frame offsets/counts ---
  auto plan = popops::embedding::plan(graph, FLOAT, numGaussians, GF, {numLookups}, {});
  Tensor table = popops::createSliceableTensor(graph, FLOAT, {numGaussians, GF},
                                               {0}, {1}, plan, {}, "g_table");
  graph.createHostWrite("g_table_h", table);

  Tensor offsets = popops::createIndicesTensor(graph, {0}, numLookups, plan, {}, "g_offsets");
  graph.createHostWrite("g_offsets_h", offsets);

  Tensor counts = graph.addVariable(UNSIGNED_INT, {numTiles}, "g_counts");
  for (unsigned t = 0; t < numTiles; ++t) graph.setTileMapping(counts[t], t);
  graph.createHostWrite("g_counts_h", counts);

  // --- gather program: multiSlice -> pin to [numTiles, perTile, GF] ---
  program::Sequence gatherProg;
  Tensor gathered = popops::multiSlice(graph, table, offsets, {0}, {1}, gatherProg, plan, {}, "g_gather");
  Tensor pinned = graph.addVariable(FLOAT, {numTiles, perTile, GF}, "g_pinned");
  for (unsigned t = 0; t < numTiles; ++t) graph.setTileMapping(pinned[t], t);
  gatherProg.add(program::Copy(gathered.reshape({numTiles, perTile, GF}), pinned));

  // --- per-tile project + blend ---
  auto projectCs = graph.addComputeSet("g_project");
  auto blendCs   = graph.addComputeSet("g_blend");

  for (unsigned t = 0; t < numTiles; ++t) {
    if (tmFb[t].empty()) continue;  // tile holds no framebuffer region

    auto localMv  = graph.clone(mvMaster,  "g_mv_"  + std::to_string(t));
    auto localMp  = graph.clone(mpMaster,  "g_mp_"  + std::to_string(t));
    auto localFxy = graph.clone(fxyMaster, "g_fxyt_" + std::to_string(t));
    graph.setTileMapping(localMv, t);
    graph.setTileMapping(localMp, t);
    graph.setTileMapping(localFxy, t);
    broadcastMvp.add(program::Copy(mvMaster, localMv));
    broadcastMvp.add(program::Copy(mpMaster, localMp));
    broadcastMvp.add(program::Copy(fxyMaster, localFxy));

    auto gaus2D = graph.addVariable(FLOAT, {perTile * sizeof(Gaussian2D)}, "g_gaus2D_" + std::to_string(t));
    auto idxs   = graph.addVariable(INT, {perTile * 2}, "g_idx_" + std::to_string(t));
    auto splat  = graph.addVariable(UNSIGNED_INT, {1}, "g_splat_" + std::to_string(t));
    auto tid    = graph.addConstant<int>(INT, {1}, {int(t)});
    graph.setTileMapping(gaus2D, t);
    graph.setTileMapping(idxs, t);
    graph.setTileMapping(splat, t);
    graph.setTileMapping(tid, t);

    auto fbSlice = paddedFramebuffer.slice(tmFb[t].front());

    auto gp = graph.addVertex(projectCs, "GatherProjectVertex");
    graph.setTileMapping(gp, t);
    graph.connect(gp["modelView"], localMv);
    graph.connect(gp["projection"], localMp);
    graph.connect(gp["tile_id"], tid);
    graph.connect(gp["fxy"], localFxy);
    graph.connect(gp["numGathered"], counts.slice(t, t + 1));
    graph.connect(gp["gathered"], pinned[t].flatten());
    graph.connect(gp["indices"], idxs);
    graph.connect(gp["gaus2D"], gaus2D);
    graph.connect(gp["splatted"], splat);

    auto bv = graph.addVertex(blendCs, "BlendVertex");
    graph.setTileMapping(bv, t);
    graph.connect(bv["gaus2D"], gaus2D);
    graph.connect(bv["splatted"], splat);
    graph.connect(bv["tile_id"], tid);
    graph.connect(bv["localFb"], fbSlice);
  }

  // --- On-device discovery (optional: replaces host computeGatherOffsets) ---
  // Must be wired up BEFORE registering broadcastMvp, since discovery adds
  // its own MVP copies to the broadcast sequence.
  if (deviceDiscovery) {
    // Shard the Gaussian table evenly across tiles for parallel projection.
    // Each tile gets ceil(numGaussians/numTiles) Gaussians. A visible Gaussian
    // typically overlaps 1-4 screen tiles, so budget ~4x assignments per Gaussian
    // in the shard as the output buffer.
    const unsigned perShard = (numGaussians + numTiles - 1) / numTiles;
    const unsigned assignCap = perShard * 4;  // max output triples per tile
    discoveryAssignCap = assignCap;
    discoveryNumTiles = numTiles;

    ipu_utils::logger()->info("Device discovery: {} gaussians / {} tiles = {}/tile, "
                              "assignCap {}", numGaussians, numTiles, perShard, assignCap);

    // Shard tensor: each tile gets perShard * GF floats from the table.
    // Pad the last tile's shard to the same size (shardCount tells the codelet
    // how many are valid).
    Tensor discTable = graph.addVariable(FLOAT, {(size_t)numTiles * perShard * GF}, "d_table");
    for (unsigned t = 0; t < numTiles; ++t) {
      auto slice = discTable.slice((size_t)t * perShard * GF, (size_t)(t + 1) * perShard * GF);
      graph.setTileMapping(slice, t);
    }
    graph.createHostWrite("d_table_h", discTable);

    // Per-tile metadata: shard offset and count.
    Tensor discOffsets = graph.addVariable(UNSIGNED_INT, {numTiles}, "d_shard_offsets");
    Tensor discCounts  = graph.addVariable(UNSIGNED_INT, {numTiles}, "d_shard_counts");
    for (unsigned t = 0; t < numTiles; ++t) {
      graph.setTileMapping(discOffsets[t], t);
      graph.setTileMapping(discCounts[t], t);
    }
    graph.createHostWrite("d_shard_offsets_h", discOffsets);
    graph.createHostWrite("d_shard_counts_h", discCounts);

    // Output: per-tile assignment triples + count.
    Tensor discAssign = graph.addVariable(UNSIGNED_INT,
        {(size_t)numTiles * assignCap * 3}, "d_assignments");
    Tensor discAssignCounts = graph.addVariable(UNSIGNED_INT, {numTiles}, "d_assign_counts");
    for (unsigned t = 0; t < numTiles; ++t) {
      auto aSlice = discAssign.slice((size_t)t * assignCap * 3, (size_t)(t + 1) * assignCap * 3);
      graph.setTileMapping(aSlice, t);
      graph.setTileMapping(discAssignCounts[t], t);
    }
    graph.createHostRead("d_assignments_h", discAssign);
    graph.createHostRead("d_assign_counts_h", discAssignCounts);

    // Wire up DiscoveryVertex on each tile.
    auto discoveryCs = graph.addComputeSet("g_discovery");
    for (unsigned t = 0; t < numTiles; ++t) {
      if (tmFb[t].empty()) continue;

      // Reuse the broadcast MVP copies already on each tile.
      auto localMvName  = "g_mv_"  + std::to_string(t);
      auto localMpName  = "g_mp_"  + std::to_string(t);
      auto localFxyName = "g_fxyt_" + std::to_string(t);

      auto shardSlice = discTable.slice((size_t)t * perShard * GF, (size_t)(t + 1) * perShard * GF);
      auto assignSlice = discAssign.slice((size_t)t * assignCap * 3, (size_t)(t + 1) * assignCap * 3);

      auto dv = graph.addVertex(discoveryCs, "DiscoveryVertex");
      graph.setTileMapping(dv, t);
      // MVP tensors were already cloned and mapped to tile t in the loop above.
      // Look them up by getting the tile mapping — but they're local variables
      // from the loop. We need to connect to the same tensors. The simplest
      // approach: store them during the per-tile loop above and reuse here.
      // Since we can't do that without restructuring, create discovery-specific
      // MVP copies from the masters (cheap — just 34 floats per tile).
      auto dMv  = graph.clone(mvMaster,  "d_mv_"  + std::to_string(t));
      auto dMp  = graph.clone(mpMaster,  "d_mp_"  + std::to_string(t));
      auto dFxy = graph.clone(fxyMaster, "d_fxy_" + std::to_string(t));
      graph.setTileMapping(dMv, t);
      graph.setTileMapping(dMp, t);
      graph.setTileMapping(dFxy, t);
      broadcastMvp.add(program::Copy(mvMaster, dMv));
      broadcastMvp.add(program::Copy(mpMaster, dMp));
      broadcastMvp.add(program::Copy(fxyMaster, dFxy));

      graph.connect(dv["modelView"], dMv);
      graph.connect(dv["projection"], dMp);
      graph.connect(dv["fxy"], dFxy);
      graph.connect(dv["gaussianShard"], shardSlice);
      graph.connect(dv["shardOffset"], discOffsets.slice(t, t + 1));
      graph.connect(dv["shardCount"], discCounts.slice(t, t + 1));
      graph.connect(dv["assignments"], assignSlice);
      graph.connect(dv["numAssignments"], discAssignCounts.slice(t, t + 1));
    }

    getPrograms().add("gather_discovery", program::Execute(discoveryCs));
  }

  // Register programs after discovery block so broadcastMvp includes all copies.
  getPrograms().add("gather_mvp", broadcastMvp);
  getPrograms().add("gather_slice", gatherProg);
  getPrograms().add("gather_project", program::Execute(projectCs));
  getPrograms().add("gather_blend", program::Execute(blendCs));
}

void IpuSplatter::executeGather(poplar::Engine& engine, const poplar::Device& device) {
  const unsigned numTiles = (unsigned)fbMapping.numTiles;
  const unsigned perTile  = gatherPerTile;
  constexpr unsigned GF   = sizeof(Gaussian3D) / sizeof(float);
  const unsigned numGaussians = (unsigned)gaussiansHost.size();

  if (!gatherInitialised) {
    gatherInitialised = true;
    enginePtr = &engine;
    // Tightly-pack the Gaussian table (15 real floats each) and upload once.
    gTableHost.resize((size_t)numGaussians * GF);
    for (size_t j = 0; j < numGaussians; ++j) {
      std::memcpy(&gTableHost[j * GF], &gaussiansHost[j], sizeof(Gaussian3D));
    }
    engine.writeTensor("g_table_h", gTableHost.data(), gTableHost.data() + gTableHost.size());

    if (deviceDiscovery) {
      // Upload the sharded Gaussian table for discovery (once).
      const unsigned perShard = (numGaussians + numTiles - 1) / numTiles;
      std::vector<float> discTableBuf((size_t)numTiles * perShard * GF, 0.f);
      std::vector<unsigned> shardOffsets(numTiles);
      std::vector<unsigned> shardCounts(numTiles);
      for (unsigned t = 0; t < numTiles; ++t) {
        unsigned begin = t * perShard;
        unsigned end = std::min(begin + perShard, numGaussians);
        shardOffsets[t] = begin;
        shardCounts[t] = (begin < numGaussians) ? (end - begin) : 0;
        for (unsigned k = begin; k < end; ++k) {
          std::memcpy(&discTableBuf[((size_t)t * perShard + (k - begin)) * GF],
                      &gaussiansHost[k], sizeof(Gaussian3D));
        }
      }
      engine.writeTensor("d_table_h", discTableBuf.data(),
                         discTableBuf.data() + discTableBuf.size());
      engine.writeTensor("d_shard_offsets_h", shardOffsets.data(),
                         shardOffsets.data() + shardOffsets.size());
      engine.writeTensor("d_shard_counts_h", shardCounts.data(),
                         shardCounts.data() + shardCounts.size());

      // Allocate readback buffers.
      discoveryAssignHost.resize((size_t)numTiles * discoveryAssignCap * 3);
      discoveryCountHost.resize(numTiles);
    }
  }

  using clk = std::chrono::steady_clock;
  auto td0 = clk::now();

  double disc_ms = 0;
  unsigned overflowDrops = 0;
  unsigned totalAssigned = 0;

  if (deviceDiscovery) {
    // Stream MVP to device (needed before discovery runs).
    engine.writeTensor("g_mv_h",  hostModelView.data(),  hostModelView.data()  + hostModelView.size());
    engine.writeTensor("g_mp_h",  hostProjection.data(), hostProjection.data() + hostProjection.size());
    engine.writeTensor("g_fxy_h", fxyHost.data(),        fxyHost.data()        + fxyHost.size());

    // Broadcast MVP to all tiles (including discovery copies).
    getPrograms().run(engine, "gather_mvp");

    // Run on-device discovery: 1440-way parallel projection.
    auto tDisc0 = clk::now();
    getPrograms().run(engine, "gather_discovery");
    auto tDisc1 = clk::now();

    // Read back assignment triples and counts.
    engine.readTensor("d_assignments_h", discoveryAssignHost.data(),
                      discoveryAssignHost.data() + discoveryAssignHost.size());
    engine.readTensor("d_assign_counts_h", discoveryCountHost.data(),
                      discoveryCountHost.data() + discoveryCountHost.size());
    auto tDiscRb = clk::now();

    // Bin assignments into per-screen-tile offset arrays (host-side, cheap).
    // Each assignment triple is (gaussian_index, dest_tile, depth_bits).
    // We keep the nearest `perTile` per dest_tile, sorted by depth.
    struct Candidate { unsigned gi; unsigned depthBits; };
    std::vector<std::vector<Candidate>> tileCands(numTiles);

    for (unsigned t = 0; t < numTiles; ++t) {
      const unsigned cnt = discoveryCountHost[t];
      const size_t base = (size_t)t * discoveryAssignCap * 3;
      for (unsigned k = 0; k < cnt; ++k) {
        unsigned gi    = discoveryAssignHost[base + k * 3 + 0];
        unsigned dest  = discoveryAssignHost[base + k * 3 + 1];
        unsigned depth = discoveryAssignHost[base + k * 3 + 2];
        if (dest < numTiles) {
          tileCands[dest].push_back({gi, depth});
        }
      }
    }

    // Build offset/count arrays for multiSlice (same format as host discovery).
    gOffsetsHost.assign((size_t)numTiles * perTile, 0u);
    gCountsHost.assign(numTiles, 0u);
    for (unsigned dest = 0; dest < numTiles; ++dest) {
      auto& cands = tileCands[dest];
      if (cands.empty()) continue;
      // Keep the nearest perTile by depth.
      if (cands.size() > perTile) {
        std::nth_element(cands.begin(), cands.begin() + perTile, cands.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.depthBits != b.depthBits) return a.depthBits < b.depthBits;
              return a.gi < b.gi;
            });
        overflowDrops += (unsigned)(cands.size() - perTile);
        cands.resize(perTile);
      }
      for (unsigned k = 0; k < cands.size(); ++k) {
        gOffsetsHost[(size_t)dest * perTile + k] = cands[k].gi;
      }
      gCountsHost[dest] = (unsigned)cands.size();
    }
    auto tBin = clk::now();

    for (unsigned c : gCountsHost) totalAssigned += c;

    // Upload the freshly computed offsets/counts.
    engine.writeTensor("g_offsets_h", gOffsetsHost.data(),
                       gOffsetsHost.data() + gOffsetsHost.size());
    engine.writeTensor("g_counts_h", gCountsHost.data(),
                       gCountsHost.data() + gCountsHost.size());

    disc_ms = std::chrono::duration<double, std::milli>(tBin - tDisc0).count();

  } else {
    // Host discovery (original path).
    GatherAssignment a = computeGatherOffsets(gaussiansHost, currentView, currentProj,
                                              currentFov, perTile);
    auto td1 = clk::now();
    disc_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    overflowDrops = a.overflowDrops;
    for (unsigned c : a.counts) totalAssigned += c;

    engine.writeTensor("g_mv_h",  hostModelView.data(),  hostModelView.data()  + hostModelView.size());
    engine.writeTensor("g_mp_h",  hostProjection.data(), hostProjection.data() + hostProjection.size());
    engine.writeTensor("g_fxy_h", fxyHost.data(),        fxyHost.data()        + fxyHost.size());
    engine.writeTensor("g_offsets_h", a.offsets.data(), a.offsets.data() + a.offsets.size());
    engine.writeTensor("g_counts_h",  a.counts.data(),  a.counts.data()  + a.counts.size());
  }

  auto td1b = clk::now();
  double stream_ms = std::chrono::duration<double, std::milli>(td1b - td0).count() - disc_ms;

  if (!deviceDiscovery) {
    // MVP broadcast only needed here for host-discovery path (device-discovery
    // already broadcast MVP before running the discovery codelet).
    getPrograms().run(engine, "gather_mvp");
  }
  auto td_mvp = clk::now();
  getPrograms().run(engine, "gather_slice");
  auto td_gather = clk::now();
  getPrograms().run(engine, "gather_project");
  auto td_proj = clk::now();
  getPrograms().run(engine, "gather_blend");
  auto td_blend = clk::now();

  engine.readTensor("g_fb_h", frameBuffer.data(), frameBuffer.data() + frameBuffer.size());
  auto td3 = clk::now();

  const double mvp_ms     = std::chrono::duration<double, std::milli>(td_mvp - td1b).count();
  const double gather_ms  = std::chrono::duration<double, std::milli>(td_gather - td_mvp).count();
  const double project_ms = std::chrono::duration<double, std::milli>(td_proj - td_gather).count();
  const double blend_ms   = std::chrono::duration<double, std::milli>(td_blend - td_proj).count();
  const double rb_ms      = std::chrono::duration<double, std::milli>(td3 - td_blend).count();
  lastTiming.compute_ms   = disc_ms + stream_ms + mvp_ms + gather_ms + project_ms + blend_ms + rb_ms;

  lastGatherTiming = {disc_ms, stream_ms, mvp_ms, gather_ms, project_ms, blend_ms, rb_ms, totalAssigned};

  static unsigned frame = 0;
  if ((frame++ % 30u) == 0u) {
    ipu_utils::logger()->info(
        "gather{}: disc {:.1f}  stream {:.1f}  mvp {:.1f}  gather {:.1f}  "
        "proj {:.1f}  blend {:.1f}  rb {:.1f}  total {:.1f}ms (drops {})",
        deviceDiscovery ? " [device-disc]" : "",
        disc_ms, stream_ms, mvp_ms, gather_ms, project_ms, blend_ms, rb_ms,
        lastTiming.compute_ms, overflowDrops);
  }
}

} // end of namespace splat
