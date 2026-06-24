// Copyright (c) 2026 Graphcore Ltd. All rights reserved.
//
// Host-side discovery for the multiSlice gather render path (branch
// jdl-experiment, Stage 2). Given the current view, project every Gaussian and
// assign it to the screen tiles its 3-sigma bounding box overlaps, producing the
// `offsets`/`counts` that drive the device gather (popops::multiSlice).
//
// CRITICAL: this uses the SAME projection math as codelets.cpp
// (ComputeCov2D, BoundingBoxFromCov, clipSpaceToViewport, the clipSize=12 guard
// band, the z>0.2 near-cull) so the set the host picks matches what the device
// would have rendered under NEWS. If the codelet math changes, change it here.
//
// Host-only (uses std::vector). It is the "discovery" piece the on-chip NEWS
// routing does implicitly; doing it on the host is the deliberate trade in
// Stage 2 ("fully on-chip -> NEWS; this speed-up -> host discovery + gather").

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include <splat/ipu_geometry.hpp>
#include <splat/viewport.hpp>
#include <tileMapping/tile_config.hpp>

namespace splat {

struct GatherAssignment {
  std::vector<unsigned> offsets;  // numTiles * perTile global indices (0-padded)
  std::vector<unsigned> counts;   // numTiles: valid entries per tile
  unsigned culled = 0;            // Gaussians not rendered anywhere (cull/guard band/offscreen)
  unsigned overflowDrops = 0;     // per-(tile,Gaussian) assignments dropped at the perTile cap
};

// Project all `gaussians` with the given view/projection and assign each to the
// tiles its 3-sigma bbox overlaps. `fovy` is the half-FOV in radians (matches
// the server's `state.fov`). `perTile` is the per-tile capacity (== the device
// gather's lookups-per-tile, e.g. 400). Gaussians is taken by non-const ref
// because Gaussian3D::ComputeCov2D is non-const.
inline GatherAssignment computeGatherOffsets(std::vector<Gaussian3D>& gaussians,
                                             const glm::mat4& viewmatrix,
                                             const glm::mat4& projmatrix,
                                             float fovy,
                                             unsigned perTile) {
  const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
  const Viewport vp(0.f, 0.f, IMWIDTH, IMHEIGHT);
  const unsigned across = (unsigned)tfb.numTilesAcross;
  const unsigned down   = (unsigned)tfb.numTilesDown;
  const unsigned numTiles = (unsigned)tfb.numTiles;

  const float tan_fovy = std::tan(fovy);
  const float tan_fovx = tan_fovy * (IMWIDTH / IMHEIGHT);
  const float focal_y  = IMHEIGHT / (2.f * tan_fovy);
  const float focal_x  = IMWIDTH  / (2.f * tan_fovx);

  const glm::mat4 mvp = projmatrix * viewmatrix;

  // Guard band: the codelet only renders a Gaussian whose bbox diagonal is
  // below tileDiag * clipSize. Replicate exactly so we don't assign Gaussians
  // the device would have dropped.
  const Bounds2f tb0 = tfb.getTileBounds(0);
  const float guardMaxDiag = tb0.diagonal().length() * 12.0f;  // clipSize == 12

  GatherAssignment out;
  out.offsets.assign((size_t)numTiles * perTile, 0u);
  out.counts.assign(numTiles, 0u);

  for (unsigned gi = 0; gi < gaussians.size(); ++gi) {
    Gaussian3D& g = gaussians[gi];
    if (g.gid <= 0.f) { continue; }  // padding slot, never a real Gaussian

    const glm::vec4 clip = mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, 1.0f);
    // Near-cull on the same value the codelet uses for g2D.z (clipSpace.z):
    if (clip.z <= 0.2f) { out.culled++; continue; }

    const glm::vec2 projMean = vp.clipSpaceToViewport(clip);
    const ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tan_fovx, tan_fovy,
                                       focal_x, focal_y);
    const ivec2 mean2D = { projMean.x, projMean.y };
    const Bounds2f bb = Gaussian2D::BoundingBoxFromCov(mean2D, cov2D);

    if (bb.diagonal().length() >= guardMaxDiag) { out.culled++; continue; }  // guard band

    int minCol = (int)std::floor(bb.min.x / IPU_TILEWIDTH);
    int maxCol = (int)std::floor(bb.max.x / IPU_TILEWIDTH);
    int minRow = (int)std::floor(bb.min.y / IPU_TILEHEIGHT);
    int maxRow = (int)std::floor(bb.max.y / IPU_TILEHEIGHT);
    if (minCol < 0) minCol = 0;
    if (minRow < 0) minRow = 0;
    if (maxCol >= (int)across) maxCol = (int)across - 1;
    if (maxRow >= (int)down)   maxRow = (int)down - 1;
    if (maxCol < minCol || maxRow < minRow) { out.culled++; continue; }  // fully offscreen

    for (int r = minRow; r <= maxRow; ++r) {
      for (int c = minCol; c <= maxCol; ++c) {
        const unsigned t = (unsigned)r * across + (unsigned)c;
        unsigned& cnt = out.counts[t];
        if (cnt < perTile) {
          out.offsets[(size_t)t * perTile + cnt] = gi;
          ++cnt;
        } else {
          // TODO(stage2): when a tile saturates, keep the perTile NEAREST by
          // clip.z instead of first-come, so dense tiles drop far Gaussians
          // (which the front-to-back blend would early-out on anyway).
          ++out.overflowDrops;
        }
      }
    }
  }
  return out;
}

}  // namespace splat
