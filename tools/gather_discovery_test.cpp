// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
//
// Host-only smoke test for splat::computeGatherOffsets (Stage 2 discovery).
// Compiles the discovery header and sanity-checks the per-tile assignment on
// synthetic Gaussians (no IPU, no PLY). Build target: gather_discovery_test.
//   ./gather_discovery_test
// Expect: non-zero total assignments spread over many tiles, no crash.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include <splat/gather_discovery.hpp>

using namespace splat;

int main() {
  const unsigned N = 2000;
  const unsigned perTile = 400;

  // Synthetic Gaussians: small, in front of an identity-view camera (+z fwd).
  std::vector<Gaussian3D> gs(N);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> xy(-2.f, 2.f), z(3.f, 7.f);
  for (unsigned i = 0; i < N; ++i) {
    gs[i].mean   = { xy(rng), xy(rng), z(rng) };
    gs[i].colour = { 0.5f, 0.5f, 0.5f, 1.0f };
    gs[i].rot    = { 1.f, 0.f, 0.f, 0.f };               // identity quaternion
    gs[i].scale  = { std::log(0.05f), std::log(0.05f), std::log(0.05f) };
    gs[i].gid    = float(i + 1);                          // gid > 0 == real
  }

  // View = identity; simple projection with clip.w = view z (perspective divide)
  // and clip.z = 0.2*z (> 0.2 so the near-cull keeps them). Column-major GLM.
  glm::mat4 view(1.0f);
  glm::mat4 proj(0.0f);
  proj[0] = glm::vec4(2.5f, 0.f, 0.f, 0.f);
  proj[1] = glm::vec4(0.f, 2.5f, 0.f, 0.f);
  proj[2] = glm::vec4(0.f, 0.f, 0.2f, 1.f);
  proj[3] = glm::vec4(0.f, 0.f, 0.f, 0.f);

  const float fovy = 0.5f;
  GatherAssignment a = computeGatherOffsets(gs, view, proj, fovy, perTile);

  unsigned total = 0, maxC = 0, nonEmpty = 0, saturated = 0;
  for (unsigned c : a.counts) {
    total += c;
    if (c > maxC) maxC = c;
    if (c > 0) ++nonEmpty;
    if (c >= perTile) ++saturated;
  }

  std::printf("gather_discovery_test\n");
  std::printf("  gaussians:        %u\n", N);
  std::printf("  tiles:            %zu\n", a.counts.size());
  std::printf("  total assignments:%u\n", total);
  std::printf("  non-empty tiles:  %u\n", nonEmpty);
  std::printf("  max per tile:     %u  (saturated tiles: %u)\n", maxC, saturated);
  std::printf("  culled:           %u\n", a.culled);
  std::printf("  overflow drops:   %u\n", a.overflowDrops);

  if (total == 0 || nonEmpty == 0) {
    std::printf("FAIL: no Gaussians were assigned to any tile.\n");
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
