// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/cpu_rasteriser.hpp>
#include <unordered_map>
#include <splat/ipu_geometry.hpp>

namespace splat {

void projectPoints(const splat::Points& in, const glm::mat4& projection, const glm::mat4& modelView,
                   std::vector<glm::vec4>& out) {
  out.resize(in.size());
  const auto mvp = projection * modelView;

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto i = 0u; i < in.size(); ++i) {
    out[i] = mvp * glm::vec4(in[i].p, 1.f);
  }
}

std::uint32_t splatPoints(cv::Mat& image,
                          const std::vector<glm::vec4>& clipCoords,
                          const splat::Points& pts,
                          const glm::mat4& projection,
                          const glm::mat4& modelView,
                          TiledFramebuffer& fb, 
                          Viewport& vp,
                          std::uint8_t value) {
  std::uint32_t count = 0u;
  const auto colour = cv::Vec3b(value, value, value);
  const auto mvp = projection * modelView;

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto i = 0u; i < pts.size(); ++i) {

    Gaussian3D g;
    auto p = pts[i].p;
    g.mean = {p.x, p.y, p.z};
    g.scale = {1.f, 1.f, 1.f};

    // auto cov3D = g.ComputeCov3D();
    // // ivec3 ComputeCov2D(const glm::mat4& projmatrix, const glm::mat4& viewmatrix, float tan_fovx, float tan_fovy)
    // auto cov2D = g.ComputeCov2D(projection, modelView, 1.0f, 1.0f);

    // printf("cov2D: %f, %f, %f\n", cov2D.x, cov2D.y, cov2D.z);



    // Convert from clip-space to pixel coords:
    glm::vec2 windowCoords = vp.clipSpaceToViewport(mvp * glm::vec4(pts[i].p, 1.f));
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < image.rows && c < image.cols) {
      image.at<cv::Vec3b>(r, c) += colour;

      #pragma omp atomic update
      count += 1;
    }
  }

  return count;
}

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const std::vector<glm::vec4>& clipCoords,
                        TiledFramebuffer& fb,
                        Viewport& vp,
                        std::uint8_t value) {
  std::uint32_t count = 0u;
  const auto colour = cv::Vec3b(value, value, value);

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto& c : counts) {
    c = 0u;
  }

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto& cs : clipCoords) {
    // Convert from clip-space to pixel coords:
    glm::vec2 windowCoords = vp.clipSpaceToViewport(cs);
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < fb.height && c < fb.width) {
      auto tidx = fb.pixCoordToTile(r, c);
      #pragma omp critical
      counts[tidx] += 1;
    }
  }
}

} // end of namespace splat
