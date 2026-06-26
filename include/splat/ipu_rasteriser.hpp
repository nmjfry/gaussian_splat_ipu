// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>
#include <chrono>
#include <thread>

#include <ipu/ipu_utils.hpp>
#include <glm/mat4x4.hpp>
#include <opencv2/imgproc.hpp>
#include <tileMapping/tile_config.hpp>


namespace splat {

struct PhaseTiming {
  double mvp_ms = 0;
  double route_ms = 0;
  double blend_ms = 0;
  double exchange_ms = 0;
  double readback_ms = 0;
  double compute_ms = 0;
  double total_ms() const { return mvp_ms + route_ms + blend_ms + exchange_ms + readback_ms; }
};

// Per-frame timing for the multiSlice gather path (Stage 2).
struct GatherTiming {
  double discovery_ms = 0;  // host: computeGatherOffsets
  double stream_ms = 0;     // host->device: mvp + offsets + counts
  double mvp_ms = 0;        // device: broadcast MVP to tiles
  double gather_ms = 0;     // device: multiSlice + copy to tile-local buffers
  double project_ms = 0;    // device: GatherProjectVertex (project + cull + sort)
  double blend_ms = 0;      // device: BlendVertex (alpha-blend)
  double readback_ms = 0;   // device->host: framebuffer
  unsigned assigned = 0;    // total (tile,Gaussian) assignments this frame
  double run_ms() const { return mvp_ms + gather_ms + project_ms + blend_ms; }
  double total_ms() const { return discovery_ms + stream_ms + run_ms() + readback_ms; }
};

// Fwd decls:
class Point3f;
typedef std::vector<Point3f> Points;
typedef std::vector<Gaussian3D> Gaussians;

class IpuSplatter : public ipu_utils::BuilderInterface {
public:
  IpuSplatter(const Points& pts, TiledFramebuffer& fb, bool noAMP);
  IpuSplatter(const Gaussians& gsns, TiledFramebuffer& fb, bool noAMP);

  virtual ~IpuSplatter() {}

  void updateProjection(const glm::mat4& mp);
  void updateModelView(const glm::mat4& mv);
  void updateFocalLengths(float fx, float fy);
  void getIPUHistogram(std::vector<u_int32_t>& counts) const;
  void getProjectedPoints(std::vector<glm::vec4>& pts) const;
  void getFrameBuffer(cv::Mat &frame) const;

  void setProfilingMode(bool enabled) { profilingMode = enabled; }
  PhaseTiming getLastPhaseTiming() const { return lastTiming; }
  GatherTiming getLastGatherTiming() const { return lastGatherTiming; }

  void broadcastMVP();
  PhaseTiming runSingleSubstep();
  void readbackCounts();
  void readbackFramebuffer();
  void readbackPhaseCycles();
  unsigned getTotalSplatCount() const;

  struct PhaseStats {
    double min_ms = 0, mean_ms = 0, max_ms = 0;
  };
  struct CycleBreakdown {
    PhaseStats clear, routing, projection, sorting, total;
  };
  CycleBreakdown getPhaseCycleStats() const;

  void setDeviceLoopMode(bool enabled) { deviceLoopMode = enabled; }
  void stopDeviceLoop();
  bool isDeviceLoopRunning() const { return deviceLoopRunning.load(); }

  // Stage 2 (branch jdl-experiment): select the multiSlice gather render path
  // instead of NEWS routing. Must be called BEFORE the graph is built
  // (i.e. before GraphManager::compileOrLoad). Default off -> NEWS, untouched.
  void setGatherMode(bool on) { gatherMode = on; }

  // Per-tile gather capacity (== device lookups/tile and blend list cap). Lower
  // = faster `run` (densest tiles bound the BSP barrier) + smaller upload, at
  // the cost of dropping the farthest Gaussians on saturated tiles. Must be set
  // before the graph is built.
  void setGatherCap(unsigned cap) { gatherPerTile = cap; }

  // Run discovery (projection + tile assignment) on the IPU instead of the
  // host CPU. Replaces the ~29ms single-threaded host loop with 1440-way
  // parallel projection on-device. Must be set before the graph is built.
  void setDeviceDiscovery(bool on) { deviceDiscovery = on; }

private:
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Gather path (host-assisted discovery + popops::multiSlice). Builds an
  // entirely separate graph so the NEWS path is unaffected.
  void buildGatherPath(poplar::Graph& graph, const poplar::Target& target);
  void executeGather(poplar::Engine& engine, const poplar::Device& device);

  ipu_utils::StreamableTensor modelView;
  ipu_utils::StreamableTensor projection;

  ipu_utils::StreamableTensor inputVertices;
  ipu_utils::StreamableTensor outputFramebuffer;
  ipu_utils::StreamableTensor counts;
  ipu_utils::StreamableTensor fxy;
  ipu_utils::StreamableTensor continueFlag;
  ipu_utils::StreamableTensor phaseCyclesStream;

  std::vector<float> hostModelView;
  std::vector<float> hostProjection;
  std::vector<float> hostVertices;
  std::vector<unsigned> splatCounts;
  std::vector<float> fxyHost;
  std::vector<int32_t> hostContinueFlag;
  std::vector<unsigned> phaseCycleData;
  TiledFramebuffer fbMapping;
  std::vector<unsigned char> frameBuffer;
  std::atomic<bool> initialised;
  const bool disableAMPVertices;
  bool profilingMode = false;
  poplar::Engine* enginePtr = nullptr;
  bool deviceLoopMode = false;
  std::atomic<bool> deviceLoopRunning{false};
  std::thread deviceThread;
  PhaseTiming lastTiming;

  // ---- Stage 2 gather-path state ----
  bool gatherMode = false;
  bool deviceDiscovery = false;        // run projection on IPU instead of host
  unsigned gatherPerTile = 400;        // device lookups per tile (== NEWS numPoints)
  bool gatherInitialised = false;
  std::vector<Gaussian3D> gaussiansHost;  // kept for per-frame host discovery
  glm::mat4 currentView{1.0f};            // last view passed to updateModelView
  glm::mat4 currentProj{1.0f};            // last projection passed to updateProjection
  float currentFov = 0.5f;                // last half-FOV (radians) from updateFocalLengths
  std::vector<float> gTableHost;          // tightly-packed Gaussian table (15 floats each)
  std::vector<unsigned> gOffsetsHost;     // per-frame offsets (numTiles*perTile)
  std::vector<unsigned> gCountsHost;      // per-frame counts (numTiles)
  GatherTiming lastGatherTiming;          // populated each executeGather frame

  // ---- Device discovery state ----
  unsigned discoveryAssignCap = 0;     // max assignments per tile (output buffer size / 3)
  unsigned discoveryNumTiles = 0;      // tiles used for discovery sharding
  std::vector<unsigned> discoveryAssignHost;  // readback: numTiles * assignCap * 3
  std::vector<unsigned> discoveryCountHost;   // readback: numTiles (assignment counts)
};

} // end of namespace splat
