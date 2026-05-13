// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>
#include <chrono>

#include <ipu/ipu_utils.hpp>
#include <glm/mat4x4.hpp>
#include <opencv2/imgproc.hpp>
#include <tileMapping/tile_config.hpp>


namespace splat {

struct PhaseTiming {
  double mvp_ms = 0;
  double compute_ms = 0;
  double exchange_ms = 0;
  double readback_ms = 0;
  double total_ms() const { return mvp_ms + compute_ms + exchange_ms + readback_ms; }
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

private:
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  ipu_utils::StreamableTensor modelView;
  ipu_utils::StreamableTensor projection;

  ipu_utils::StreamableTensor inputVertices;
  ipu_utils::StreamableTensor outputFramebuffer;
  ipu_utils::StreamableTensor counts;
  ipu_utils::StreamableTensor fxy;

  std::vector<float> hostModelView;
  std::vector<float> hostProjection;
  std::vector<float> hostVertices;
  std::vector<unsigned> splatCounts;
  std::vector<float> fxyHost;
  TiledFramebuffer fbMapping;
  std::vector<unsigned char> frameBuffer;
  std::atomic<bool> initialised;
  const bool disableAMPVertices;
  bool profilingMode = false;
  PhaseTiming lastTiming;
};

} // end of namespace splat
