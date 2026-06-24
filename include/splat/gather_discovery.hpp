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

#include <algorithm>
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

  // Per-tile candidate buffers. Pass 1 (parallel over Gaussians) appends every
  // (tile, Gaussian) candidate with its depth; pass 2 (parallel over tiles)
  // keeps the perTile NEAREST by (depth, gid). Because the kept set is chosen
  // by depth — not arrival order — the result is DETERMINISTIC frame-to-frame
  // even though pass 1 runs in parallel. That removes the saturated-tile
  // flicker the first-come-drop version had, and drops far Gaussians (which the
  // front-to-back blend early-outs on) rather than arbitrary ones.
  //
  // Cmax bounds the per-tile candidate storage. A tile with more than Cmax
  // candidates loses the surplus by arrival order (non-deterministic) — but
  // Cmax = 3*perTile is far above realistic densities, so this is a safety
  // valve, not the normal path. candDepth/candGi are left uninitialised on
  // purpose (only [0, count) is written/read).
  const unsigned Cmax = perTile * 3u;
  std::vector<unsigned> candCount(numTiles, 0u);
  std::vector<float>    candDepth((size_t)numTiles * Cmax);
  std::vector<unsigned> candGi((size_t)numTiles * Cmax);

  unsigned culled = 0;
  const int n = (int)gaussians.size();
  #pragma omp parallel for schedule(static) reduction(+ : culled)
  for (int gi = 0; gi < n; ++gi) {
    Gaussian3D& g = gaussians[gi];
    if (g.gid <= 0.f) { continue; }  // padding slot, never a real Gaussian

    const glm::vec4 clip = mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, 1.0f);
    if (clip.z <= 0.2f) { ++culled; continue; }  // near-cull (== codelet g2D.z)

    const glm::vec2 projMean = vp.clipSpaceToViewport(clip);
    // Cheap frustum reject BEFORE the expensive ComputeCov2D: a Gaussian whose
    // mean is more than guardMaxDiag px off-screen cannot have its bbox overlap
    // the screen (a larger bbox would be guard-band-culled anyway). Skips the
    // costly cov2D for the many off-to-the-side Gaussians on a zoomed view.
    if (projMean.x < -guardMaxDiag || projMean.x > IMWIDTH  + guardMaxDiag ||
        projMean.y < -guardMaxDiag || projMean.y > IMHEIGHT + guardMaxDiag) {
      ++culled; continue;
    }

    const ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tan_fovx, tan_fovy,
                                       focal_x, focal_y);
    const ivec2 mean2D = { projMean.x, projMean.y };
    const Bounds2f bb = Gaussian2D::BoundingBoxFromCov(mean2D, cov2D);

    if (bb.diagonal().length() >= guardMaxDiag) { ++culled; continue; }  // guard band

    int minCol = (int)std::floor(bb.min.x / IPU_TILEWIDTH);
    int maxCol = (int)std::floor(bb.max.x / IPU_TILEWIDTH);
    int minRow = (int)std::floor(bb.min.y / IPU_TILEHEIGHT);
    int maxRow = (int)std::floor(bb.max.y / IPU_TILEHEIGHT);
    if (minCol < 0) minCol = 0;
    if (minRow < 0) minRow = 0;
    if (maxCol >= (int)across) maxCol = (int)across - 1;
    if (maxRow >= (int)down)   maxRow = (int)down - 1;
    if (maxCol < minCol || maxRow < minRow) { ++culled; continue; }  // fully offscreen

    for (int r = minRow; r <= maxRow; ++r) {
      for (int c = minCol; c <= maxCol; ++c) {
        const unsigned t = (unsigned)r * across + (unsigned)c;
        unsigned slot;
        #pragma omp atomic capture
        { slot = candCount[t]; candCount[t] += 1u; }
        if (slot < Cmax) {
          candDepth[(size_t)t * Cmax + slot] = clip.z;
          candGi[(size_t)t * Cmax + slot]    = (unsigned)gi;
        }
      }
    }
  }

  // Pass 2: per tile keep the perTile nearest candidates (deterministic).
  unsigned overflow = 0;
  #pragma omp parallel for schedule(dynamic, 16) reduction(+ : overflow)
  for (int t = 0; t < (int)numTiles; ++t) {
    const unsigned cnt = std::min(candCount[t], Cmax);
    if (cnt == 0) continue;
    const size_t base = (size_t)t * Cmax;
    const unsigned keep = std::min(cnt, perTile);

    // Order candidate indices by (depth asc, gid asc) so the kept set is the
    // nearest perTile and the choice is independent of pass-1 arrival order.
    std::vector<unsigned> idx(cnt);
    for (unsigned k = 0; k < cnt; ++k) idx[k] = k;
    auto cmp = [&](unsigned x, unsigned y) {
      const float dx = candDepth[base + x], dy = candDepth[base + y];
      if (dx != dy) return dx < dy;
      return candGi[base + x] < candGi[base + y];  // deterministic tie-break
    };
    if (cnt > keep) {
      std::nth_element(idx.begin(), idx.begin() + keep, idx.end(), cmp);
      overflow += (cnt - keep);
    }
    for (unsigned k = 0; k < keep; ++k) {
      out.offsets[(size_t)t * perTile + k] = candGi[base + idx[k]];
    }
    out.counts[t] = keep;
  }

  out.culled = culled;
  out.overflowDrops = overflow;
  return out;
}

}  // namespace splat
