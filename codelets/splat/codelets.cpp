// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/tileMapping/tile_config.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/viewport.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/ipu_geometry.hpp>

using namespace splat;

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>

inline float ipu_exp(float x) { return __builtin_ipu_exp(x); }
inline float ipu_max(float a, float b) { return __builtin_ipu_max(a, b); }
inline float ipu_min(float a, float b) { return __builtin_ipu_min(a, b); }
inline float ipu_clamp(float x, float lo, float hi) {
  return __builtin_ipu_min(__builtin_ipu_max(x, lo), hi);
}
#else
inline float ipu_exp(float x) { return expf(x); }
inline float ipu_max(float a, float b) { return a > b ? a : b; }
inline float ipu_min(float a, float b) { return a < b ? a : b; }
inline float ipu_clamp(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
#endif

template <typename G, typename Vec> bool insertAt(Vec &buffer, unsigned idx, const G& g) {
    if (idx + sizeof(g) > buffer.size()) {
      return false;
    }
    memcpy(&buffer[idx], &g, sizeof(g));
    return true;
  }

// G must have a float gid as the last element
// return false only if the buffer is full
template <typename G, typename Vec> bool insert(Vec &buffer, const G& g) {
  unsigned idx = buffer.size();
  for (auto i = 0; i < buffer.size(); i+=sizeof(g)) {
    float gid;
    // assumes gid is float and last element in the struct
    size_t gidIdx = (sizeof(g) - sizeof(gid)) / sizeof(float);
    memcpy(&gid, &buffer[i+gidIdx], sizeof(gid));
    if (gid == g.gid) {
      // stop since the gaussian already is in the buffer
      return true;
    }
    if (gid == 0u && i < idx) {
      idx = i;
    }
  }
  return insertAt<G, Vec>(buffer, idx, g);
}

template<typename G, typename Vec> G unpack(Vec &buffer, unsigned idx) {
  G g;
  memcpy(&g, &buffer[idx], sizeof(g));
  return g;
}

// invalidate a gaussian in the buffer
template<typename G, typename Vec> void evict(Vec &buffer, unsigned idx) {
  G g;
  g.gid = 0.f;
  insertAt(buffer, idx, g);
}

struct ProjParams {
  glm::mat4 mvp;
  glm::mat4 projmatrix;
  glm::mat4 viewmatrix;
  glm::vec2 tanfov;
  glm::vec2 focal;
};

// Bloom routing mode (A/B flag — flip and rebuild to compare flicker):
//  * true  : "flow-through" — a Gaussian that blooms into a non-anchor tile is
//            rendered on arrival in readInput and then dropped. The anchor tile
//            re-blooms every frame so the extent is sustained. This avoids the
//            ping-pong where every bloomed tile parks the copy in vertsIn and
//            then ships it back toward the anchor, which scales with the bloom
//            area (~Gaussian size^2) and saturates the NEWS channels -> flicker.
//  * false : original behaviour (store bloom copies in vertsIn, re-send toward
//            anchor in projectAndRoute).
// Default OFF so the baseline matches the original fast path; flip to true to
// A/B the flicker vs the (small) extra per-frame render cost.
static constexpr bool kBloomFlowThrough = false;


// Single-worker vertex: clears output buffers, reads NEWS input channels,
// projects Gaussians, routes them via Manhattan routing, builds a sorted
// Gaussian2D list for the blend phase.
class RouteVertex : public poplar::Vertex {

public:
  poplar::Input<poplar::Vector<float>> modelView;
  poplar::Input<poplar::Vector<float>> projection;

  poplar::Input<poplar::Vector<int>> tile_id;
  poplar::Input<poplar::Vector<float>> fxy;
  poplar::Output<poplar::Vector<unsigned>> splatted;
  poplar::Output<poplar::Vector<unsigned>> phaseCycles;

  poplar::InOut<poplar::Vector<float>> vertsIn;
  poplar::Output<poplar::Vector<int>> indices;
  poplar::Output<poplar::Vector<float>> gaus2D;

  poplar::Input<poplar::Vector<float>> rightIn;
  poplar::Output<poplar::Vector<float>> rightOut;

  poplar::Input<poplar::Vector<float>> leftIn;
  poplar::Output<poplar::Vector<float>> leftOut;

  poplar::Input<poplar::Vector<float>> upIn;
  poplar::Output<poplar::Vector<float>> upOut;

  poplar::Input<poplar::Vector<float>> downIn;
  poplar::Output<poplar::Vector<float>> downOut;

  float clipSize;

  template<typename G> bool send(const G &g, directions dirs) {
    bool sent = true;
    if (dirs.right) {
      sent = sent && insert(rightOut, g);
    }
    if (dirs.left) {
      sent = sent && insert(leftOut, g);
    }
    if (dirs.up) {
      sent = sent && insert(upOut, g);
    }
    if (dirs.down) {
      sent = sent && insert(downOut, g);
    }
    return sent;
  }

  template<typename G> bool sendOnce(const G &g, direction dir) {
    if (dir == direction::right) {
      return insert(rightOut, g);
    } else if (dir == direction::down) {
      return insert(downOut, g);
    } else if (dir == direction::left) {
      return insert(leftOut, g);
    } else if (dir == direction::up) {
      return insert(upOut, g);
    }
    return false;
  }

  /// Protocol for sending a gaussian to a neighbouring tile
  /// spreads out left and right from centre in 2 beams,
  /// then sends up and down from these beams:
  ///         |||||||||||
  ///         <----o---->
  ///         |||||||||||
  /// currently sends back at edges, so we render twice...
  template<typename G> bool protocol(const G& g, const directions& sendTo, const direction& recievedFrom) {
    if (recievedFrom == direction::right && sendTo.left) {
      bool ok = sendOnce(g, direction::left);
      if (sendTo.down) {
        ok = ok && sendOnce(g, direction::down);
      }
      if (sendTo.up) {
        ok = ok && sendOnce(g, direction::up);
      }
      return ok;
    }

    if (recievedFrom == direction::left && sendTo.right) {
      bool ok = sendOnce(g, direction::right);
      if (sendTo.down) {
        ok = ok && sendOnce(g, direction::down);
      }
      if (sendTo.up) {
        ok = ok && sendOnce(g, direction::up);
      }
      return ok;
    }

    if (recievedFrom == direction::up && sendTo.down) {
      return sendOnce(g, direction::down);
    }

    if (recievedFrom == direction::down && sendTo.up) {
      return sendOnce(g, direction::up);
    }

    if (sendTo.any()) {
      bool ok = true;
      if (sendTo.up && recievedFrom != direction::up) {
        ok = ok && sendOnce(g, direction::up);
      }
      if (sendTo.down && recievedFrom != direction::down) {
        ok = ok && sendOnce(g, direction::down);
      }
      return ok;
    }
    return false;
  }

  template <typename G>
  void swap(float *a, float *b) {
    G temp;
    std::memcpy(&temp, a, sizeof(G));
    std::memcpy(a, b, sizeof(G));
    std::memcpy(b, &temp, sizeof(G));
  }

  template <typename G>
  int partition(float *elements, int low, int high) {
    high = high * sizeof(G);
    low = low * sizeof(G);
    G pivotG;
    std::memcpy(&pivotG, &elements[high], sizeof(G));
    int i = low - (int)sizeof(G);
    for (int j = low; j <= high - (int)sizeof(G); j+=sizeof(G)) {
        G gm;
        std::memcpy(&gm, &elements[j], sizeof(G));
        // Front-to-back: in COLMAP view space, closest = smallest (positive) z.
        // Sort ASCENDING so front Gaussians composite first.
        if (gm.z <= pivotG.z) {
            i+=sizeof(G);
            swap<G>(&elements[i], &elements[j]);
        }
    }
    swap<G>(&elements[i + (int)sizeof(G)], &elements[high]);
    return (i + (int)sizeof(G)) / sizeof(G);
  }

  template <typename G>
  void iterativeQuickSort(float *elements, int l, int h) {
      int top = -1;
      indices[++top] = l;
      indices[++top] = h;
      while (top >= 0) {
          h = indices[top--];
          l = indices[top--];

          int pi = partition<G>(elements, l, h);

          if (pi - 1 > l) {
              indices[++top] = l;
              indices[++top] = pi - 1;
          }

          if (pi + 1 < h) {
              indices[++top] = pi + 1;
              indices[++top] = h;
          }
      }
  }

  // Quicksort the first `count` Gaussian2D in `buffer`, ascending by z
  // (front-to-back). O(n log n) on the unsorted-each-frame lists we see here
  // (insertion sort was O(n^2) because the list is built in vertsIn storage
  // order, not depth order). Sorts exactly [0, count): the previous code passed
  // `count` as the inclusive high index, pulling one empty slot into the range.
  template<typename G>
  void sortBuffer(poplar::Vector<float>& buffer, unsigned count) {
    if (count < 2) {
      return;
    }
    if (count > indices.size()) {
      return;  // safety: the toRender clamp keeps count <= capacity
    }
    for (auto i = 0u; i < indices.size(); ++i) {
      indices[i] = 0;
    }
    iterativeQuickSort<G>(&buffer[0], 0, (int)count - 1);
  }

  template<typename G, typename Vec> void clearGidOnly(Vec &buffer) {
    constexpr size_t gidOffset = (sizeof(G) - sizeof(float)) / sizeof(float);
    float zero = 0.f;
    for (auto i = 0u; i < buffer.size(); i += sizeof(G)) {
      memcpy(&buffer[i + gidOffset], &zero, sizeof(float));
    }
  }

  void clearOutBuffers() {
    clearGidOnly<Gaussian3D>(rightOut);
    clearGidOnly<Gaussian3D>(leftOut);
    clearGidOnly<Gaussian3D>(upOut);
    clearGidOnly<Gaussian3D>(downOut);
  }

  template<typename InternalStorage> unsigned projectAndRoute(InternalStorage& buffer,
                                                              const ProjParams& pp,
                                                              const TiledFramebuffer& tfb,
                                                              const splat::Viewport& vp,
                                                              unsigned toRender) {
    const auto tb = tfb.getTileBounds(tile_id[0]);

    for (auto i = 0; i < buffer.size(); i+=sizeof(Gaussian3D)) {

      Gaussian3D g = unpack<Gaussian3D>(buffer, i);
      auto scale = g.scale;
      if (g.gid <= 0) {
        continue;
      }

      auto clipSpace = pp.mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, 1.0f);
      auto projMean = vp.clipSpaceToViewport(clipSpace);

      // Scale used as-is (matching original 3DGS — no lambda division)
      // render and clip, send to the halo region around the current tile
      ivec3 cov2D = g.ComputeCov2D(pp.projmatrix, pp.viewmatrix, pp.tanfov.x, pp.tanfov.y, pp.focal.x, pp.focal.y);
      ivec2 projMean2D = {projMean.x, projMean.y};
      auto bb = Gaussian2D::BoundingBoxFromCov(projMean2D, cov2D);
      Gaussian2D g2D(projMean2D, g.colour, cov2D, clipSpace.z);
      g.scale = scale;

      bool withinGuardBand = bb.diagonal().length() < tb.diagonal().length() * clipSize;

      directions dirs;
      if (withinGuardBand) {
        bb = bb.clip(tb, dirs);
      }


      bool ok = true;
      if (tb.contains(g2D.mean)) {
        ok = send(g, dirs);
      } else {
        // evict and send on to the next tile
        auto dstTile = tfb.pixCoordToTile(projMean.y, projMean.x);
        auto dstCentre = tfb.getTileBounds(dstTile).centroid();
        auto direction = tfb.getBestDirection(tb.centroid(), dstCentre);
        evict<Gaussian3D>(buffer, i);
        if (!sendOnce(g, direction)) {
          // guard against losing a gaussian, put it right back in the buffer
          insertAt(vertsIn, i, g);
        }
      }

      // COLMAP convention: visible Gaussians have view_z > 0 (in front of camera).
      // Also near-cull anything with view_z below a small positive threshold.
      if (withinGuardBand && g2D.z > 0.2f) {
        auto g2Idx = toRender * sizeof(Gaussian2D);
        // Only count it if it actually fit: otherwise numToRender would run past
        // the gaus2D capacity, the sort would be skipped and BlendVertex would
        // read out of bounds (a flicker source on dense/large-Gaussian tiles).
        if (insertAt(gaus2D, g2Idx, g2D)) {
          toRender++;
        }
      }
    }

    return toRender;
  }

  void readInput(poplar::Input<poplar::Vector<float>> &bufferIn,
                                    const direction& recievedFrom,
                                    const ProjParams& pp,
                                    const TiledFramebuffer& tfb,
                                    const splat::Viewport& vp,
                                    unsigned& toRender) {
    // Get the boundary of the current tile's framebuffer section
    const auto tb = tfb.getTileBounds(tile_id[0]);
    const auto tbPrev = tfb.getTileBounds(tfb.getNearbyTile(tile_id[0], recievedFrom));

    // Iterate over the input channel and unpack the Gaussian3D structs
    for (auto i = 0; i < bufferIn.size(); i+=sizeof(Gaussian3D)) {
      Gaussian3D g = unpack<Gaussian3D>(bufferIn, i);
      auto scale = g.scale;
      if (g.gid <= 0) {
        // gid 0 if the place in the buffer is not occupied,
        // since the channels are filled from the front we can break
        // when we hit an empty slot
        continue;
      }

      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, 1.0f};
      auto clipSpace = pp.mvp * glmMean;
      auto projMean = vp.clipSpaceToViewport(clipSpace);

      if (tb.contains(ivec2{projMean.x, projMean.y})) {
        // anchor arrived so we insert and let
        // render main handle the rest
        bool overflow = !insert(vertsIn, g);
        continue;
      }

      // Scale used as-is (matching original 3DGS — no lambda division)

      ivec3 cov2D = g.ComputeCov2D(pp.projmatrix, pp.viewmatrix, pp.tanfov.x, pp.tanfov.y, pp.focal.x, pp.focal.y);
      ivec2 projMean2D = {projMean.x, projMean.y};
      Gaussian2D g2D(projMean2D, g.colour, cov2D, clipSpace.z);
      g.scale = scale;

      auto dstTile = tfb.pixCoordToTile(g2D.mean.y, g2D.mean.x);
      // dstTile = dstTile < 0 ? 0 : dstTile;
      ivec2 dstCentre = tfb.getTileBounds(dstTile).centroid();
      ivec2 prevCentre = tbPrev.centroid();
      ivec2 curCentre = tb.centroid();

      auto prevDist = tfb.manhattanDistance(prevCentre, dstCentre);
      auto curDist = tfb.manhattanDistance(curCentre, dstCentre);

      if (curDist < prevDist) {
        auto direction = tfb.getBestDirection(curCentre, dstCentre);
        if (!sendOnce(g, direction)) {
          // guard against losing a gaussian
          // we get here if the out buffer is full but the
          // gaussian is in transit to another tile
        }
        bool overflow = !insert(vertsIn, g);
        continue;
      }

      // the gaussian is being propagated away from the anchor (bloom),
      // we need to render and pass it on until the extent is fully rendered.
      auto bb = Gaussian2D::BoundingBoxFromCov(projMean2D, cov2D);

      if (bb.diagonal().length() < tb.diagonal().length() * clipSize) {
        directions sendTo;
        auto clippedBB = bb.clip(tb, sendTo);
        protocol<Gaussian3D>(g, sendTo, recievedFrom);

        if (kBloomFlowThrough && g2D.z > 0.2f) {
          // Flow-through: render the bloom copy here on arrival rather than
          // parking it in vertsIn (which would then be shipped back toward the
          // anchor every frame). The anchor re-blooms each frame, so the front
          // stays populated without the ping-pong traffic.
          auto g2Idx = toRender * sizeof(Gaussian2D);
          if (insertAt(gaus2D, g2Idx, g2D)) {
            toRender++;
          }
        }
      }

      if (!kBloomFlowThrough) {
        bool overflow = !insert(vertsIn, g);
      }
    }
  }

  bool compute() {
#ifdef __IPU__
    unsigned cyc0 = __builtin_ipu_get_scount_l();
#endif

    clearOutBuffers();

    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    clipSize = 12.0f;

    const auto viewmatrix = glm::transpose(glm::make_mat4(&modelView[0]));
    const auto projmatrix = glm::transpose(glm::make_mat4(&projection[0]));

    float tan_fovy = glm::tan(fxy[0]);
    float tan_fovx = tan_fovy * (float(tfb.width) / float(tfb.height));
    float focal_y = float(tfb.height) / (2.f * tan_fovy);
    float focal_x = float(tfb.width)  / (2.f * tan_fovx);

    ProjParams pp;
    pp.mvp = projmatrix * viewmatrix;
    pp.projmatrix = projmatrix;
    pp.viewmatrix = viewmatrix;
    pp.tanfov = {tan_fovx, tan_fovy};
    pp.focal = {focal_x, focal_y};

#ifdef __IPU__
    unsigned cyc1 = __builtin_ipu_get_scount_l();
#endif

    // toRender accumulates across the input channels (bloom flow-through, when
    // enabled) and then projectAndRoute (anchor-owned + in-transit Gaussians).
    unsigned numToRender = 0;
    readInput(rightIn, direction::right, pp, tfb, vp, numToRender);
    readInput(leftIn, direction::left, pp, tfb, vp, numToRender);
    readInput(upIn, direction::up, pp, tfb, vp, numToRender);
    readInput(downIn, direction::down, pp, tfb, vp, numToRender);

#ifdef __IPU__
    unsigned cyc2 = __builtin_ipu_get_scount_l();
#endif

    numToRender = projectAndRoute(vertsIn, pp, tfb, vp, numToRender);

#ifdef __IPU__
    unsigned cyc3 = __builtin_ipu_get_scount_l();
#endif

    if (numToRender > 1) {
      sortBuffer<Gaussian2D>(gaus2D, numToRender);
    }
    splatted[0] = numToRender;

#ifdef __IPU__
    unsigned cyc4 = __builtin_ipu_get_scount_l();
    phaseCycles[0] = cyc1 - cyc0; // clear + setup
    phaseCycles[1] = cyc2 - cyc1; // routing (readInput x4)
    phaseCycles[2] = cyc3 - cyc2; // projection (projectAndRoute)
    phaseCycles[3] = cyc4 - cyc3; // sorting
    phaseCycles[4] = cyc4 - cyc0; // total
#endif

    return true;
  }

};


// On-device discovery for the gather path. Each tile projects a shard of the
// full Gaussian set and outputs (global_index, dest_tile_id, depth) triples
// for every visible (tile, Gaussian) assignment. The host reads these back,
// bins them into per-tile offset arrays, and feeds them to multiSlice.
// Replaces the 29ms host-side computeGatherOffsets with ~1440-way parallel
// projection on the IPU.
class DiscoveryVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float>> modelView;
  poplar::Input<poplar::Vector<float>> projection;
  poplar::Input<poplar::Vector<float>> fxy;
  poplar::Input<poplar::Vector<float>> gaussianShard;   // this tile's slice of the table
  poplar::Input<poplar::Vector<unsigned>> shardOffset;   // global index of first Gaussian
  poplar::Input<poplar::Vector<unsigned>> shardCount;    // number of valid Gaussians in shard
  // Each assignment is 3 unsigned: (global_gaussian_index, dest_tile, depth_bits).
  // depth_bits = float-as-uint for deterministic nearest-first selection on host.
  poplar::Output<poplar::Vector<unsigned>> assignments;
  poplar::Output<poplar::Vector<unsigned>> numAssignments;

  bool compute() {
    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    const unsigned across = tfb.numTilesAcross;
    const unsigned down   = tfb.numTilesDown;
    constexpr float clipSize = 12.0f;

    const auto viewmatrix = glm::transpose(glm::make_mat4(&modelView[0]));
    const auto projmatrix = glm::transpose(glm::make_mat4(&projection[0]));
    const float tan_fovy = glm::tan(fxy[0]);
    const float tan_fovx = tan_fovy * (float(IMWIDTH) / float(IMHEIGHT));
    const float focal_y = float(IMHEIGHT) / (2.f * tan_fovy);
    const float focal_x = float(IMWIDTH)  / (2.f * tan_fovx);
    const glm::mat4 mvp = projmatrix * viewmatrix;

    const Bounds2f tb0 = tfb.getTileBounds(0);
    const float guardMaxDiag = tb0.diagonal().length() * clipSize;

    constexpr unsigned GF = sizeof(Gaussian3D) / sizeof(float);
    const unsigned n = shardCount[0];
    const unsigned base = shardOffset[0];
    const unsigned maxOut = assignments.size() / 3u;
    unsigned count = 0;

    for (unsigned k = 0; k < n; ++k) {
      Gaussian3D g;
      std::memcpy(&g, &gaussianShard[k * GF], sizeof(g));
      if (g.gid <= 0.f) continue;

      auto clipSpace = mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, 1.0f);
      if (clipSpace.z <= 0.2f) continue;

      auto projMean = vp.clipSpaceToViewport(clipSpace);
      if (projMean.x < -guardMaxDiag || projMean.x > IMWIDTH  + guardMaxDiag ||
          projMean.y < -guardMaxDiag || projMean.y > IMHEIGHT + guardMaxDiag)
        continue;

      ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tan_fovx, tan_fovy,
                                    focal_x, focal_y);
      ivec2 mean2D = {projMean.x, projMean.y};
      auto bb = Gaussian2D::BoundingBoxFromCov(mean2D, cov2D);
      if (bb.diagonal().length() >= guardMaxDiag) continue;

      int minCol = (int)floor(bb.min.x / IPU_TILEWIDTH);
      int maxCol = (int)floor(bb.max.x / IPU_TILEWIDTH);
      int minRow = (int)floor(bb.min.y / IPU_TILEHEIGHT);
      int maxRow = (int)floor(bb.max.y / IPU_TILEHEIGHT);
      if (minCol < 0) minCol = 0;
      if (minRow < 0) minRow = 0;
      if (maxCol >= (int)across) maxCol = (int)across - 1;
      if (maxRow >= (int)down)   maxRow = (int)down - 1;
      if (maxCol < minCol || maxRow < minRow) continue;

      unsigned depthBits;
      float z = clipSpace.z;
      std::memcpy(&depthBits, &z, sizeof(unsigned));

      for (int r = minRow; r <= maxRow; ++r) {
        for (int c = minCol; c <= maxCol; ++c) {
          if (count >= maxOut) goto done;
          unsigned t = (unsigned)r * across + (unsigned)c;
          assignments[count * 3 + 0] = base + k;
          assignments[count * 3 + 1] = t;
          assignments[count * 3 + 2] = depthBits;
          count++;
        }
      }
    }
    done:
    numAssignments[0] = count;
    return true;
  }
};


// Single-worker vertex for the multiSlice gather render path (branch
// jdl-experiment, Stage 2). The host has already done discovery + routing:
// `gathered` holds exactly this tile's Gaussians (those whose 3-sigma bbox
// overlaps it), tightly packed at sizeof(Gaussian3D)/sizeof(float) floats each
// (NOT the 60-float stride vertsIn uses). This vertex just projects them,
// collects the visible subset into `gaus2D`, and sorts — no NEWS, no routing.
// It writes gaus2D exactly like RouteVertex so BlendVertex is unchanged.
class GatherProjectVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float>> modelView;
  poplar::Input<poplar::Vector<float>> projection;
  poplar::Input<poplar::Vector<int>> tile_id;
  poplar::Input<poplar::Vector<float>> fxy;
  poplar::Input<poplar::Vector<unsigned>> numGathered;  // valid Gaussians for this tile
  poplar::Input<poplar::Vector<float>> gathered;        // perTile * (sizeof(G)/4) floats
  poplar::Output<poplar::Vector<int>> indices;          // sort scratch
  poplar::Output<poplar::Vector<float>> gaus2D;
  poplar::Output<poplar::Vector<unsigned>> splatted;

  float clipSize;

  // --- sort helpers (duplicated from RouteVertex to avoid touching it) ---
  template <typename G>
  void swap(float *a, float *b) {
    G temp;
    std::memcpy(&temp, a, sizeof(G));
    std::memcpy(a, b, sizeof(G));
    std::memcpy(b, &temp, sizeof(G));
  }

  template <typename G>
  int partition(float *elements, int low, int high) {
    high = high * sizeof(G);
    low = low * sizeof(G);
    G pivotG;
    std::memcpy(&pivotG, &elements[high], sizeof(G));
    int i = low - (int)sizeof(G);
    for (int j = low; j <= high - (int)sizeof(G); j += sizeof(G)) {
      G gm;
      std::memcpy(&gm, &elements[j], sizeof(G));
      if (gm.z <= pivotG.z) {
        i += sizeof(G);
        swap<G>(&elements[i], &elements[j]);
      }
    }
    swap<G>(&elements[i + (int)sizeof(G)], &elements[high]);
    return (i + (int)sizeof(G)) / sizeof(G);
  }

  template <typename G>
  void iterativeQuickSort(float *elements, int l, int h) {
    int top = -1;
    indices[++top] = l;
    indices[++top] = h;
    while (top >= 0) {
      h = indices[top--];
      l = indices[top--];
      int pi = partition<G>(elements, l, h);
      if (pi - 1 > l) { indices[++top] = l; indices[++top] = pi - 1; }
      if (pi + 1 < h) { indices[++top] = pi + 1; indices[++top] = h; }
    }
  }

  template <typename G>
  void sortBuffer(poplar::Vector<float>& buffer, unsigned count) {
    if (count < 2) { return; }
    if (count > indices.size()) { return; }
    for (auto i = 0u; i < indices.size(); ++i) { indices[i] = 0; }
    iterativeQuickSort<G>(&buffer[0], 0, (int)count - 1);
  }

  bool compute() {
    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    clipSize = 12.0f;

    const auto viewmatrix = glm::transpose(glm::make_mat4(&modelView[0]));
    const auto projmatrix = glm::transpose(glm::make_mat4(&projection[0]));

    float tan_fovy = glm::tan(fxy[0]);
    float tan_fovx = tan_fovy * (float(tfb.width) / float(tfb.height));
    float focal_y = float(tfb.height) / (2.f * tan_fovy);
    float focal_x = float(tfb.width)  / (2.f * tan_fovx);
    const glm::mat4 mvp = projmatrix * viewmatrix;

    // Gathered Gaussians are tightly packed (no 60-float stride waste):
    constexpr unsigned GF = sizeof(Gaussian3D) / sizeof(float);
    const auto tb = tfb.getTileBounds(tile_id[0]);
    const unsigned n = numGathered[0];

    unsigned toRender = 0;
    for (unsigned k = 0; k < n; ++k) {
      Gaussian3D g;
      std::memcpy(&g, &gathered[k * GF], sizeof(g));
      if (g.gid <= 0) { continue; }

      auto clipSpace = mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, 1.0f);
      auto projMean = vp.clipSpaceToViewport(clipSpace);
      ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tan_fovx, tan_fovy, focal_x, focal_y);
      ivec2 projMean2D = {projMean.x, projMean.y};
      auto bb = Gaussian2D::BoundingBoxFromCov(projMean2D, cov2D);
      Gaussian2D g2D(projMean2D, g.colour, cov2D, clipSpace.z);

      bool withinGuardBand = bb.diagonal().length() < tb.diagonal().length() * clipSize;
      if (withinGuardBand && g2D.z > 0.2f) {
        auto g2Idx = toRender * sizeof(Gaussian2D);
        if (insertAt(gaus2D, g2Idx, g2D)) { toRender++; }
      }
    }

    if (toRender > 1) { sortBuffer<Gaussian2D>(gaus2D, toRender); }
    splatted[0] = toRender;
    return true;
  }
};


// Multi-worker vertex: clears framebuffer and alpha-blends the sorted
// Gaussian2D list produced by RouteVertex. Pixel rows are distributed
// across all 6 IPU workers for ~6x parallel speedup on the hot loop.
class BlendVertex : public poplar::MultiVertex {

public:
  poplar::Input<poplar::Vector<float>> gaus2D;
  poplar::Input<poplar::Vector<unsigned>> splatted;
  poplar::Input<poplar::Vector<int>> tile_id;
  poplar::Output<poplar::Vector<unsigned char>> localFb;

  unsigned toByteBufferIndex(float x, float y) {
    return unsigned(x + y * IPU_TILEWIDTH) * 4;
  }

  static unsigned char clampToU8(float v) {
    return (unsigned char)ipu_clamp(v * 255.0f, 0.0f, 255.0f);
  }

  // Pack RGBX into a single 32-bit word to avoid byte-level
  // race conditions between workers on the same IPU tile.
  void setPixel(float x, float y, const ivec4 &colour) {
    unsigned idx = toByteBufferIndex(x, y);
    // Read existing pixel as 32-bit word
    unsigned word;
    memcpy(&word, &localFb[idx], 4);
    // Unpack, add, clamp
    unsigned char r0 = word & 0xFF;
    unsigned char g0 = (word >> 8) & 0xFF;
    unsigned char b0 = (word >> 16) & 0xFF;
    unsigned r = (unsigned)r0 + clampToU8(colour.x);
    unsigned g = (unsigned)g0 + clampToU8(colour.y);
    unsigned b = (unsigned)b0 + clampToU8(colour.z);
    word = (r > 255 ? 255 : r)
         | ((g > 255 ? 255 : g) << 8)
         | ((b > 255 ? 255 : b) << 16);
    memcpy(&localFb[idx], &word, 4);
  }

  ivec2 viewspaceToTile(const ivec2& pt, ivec2 tlBound) {
    return {floor(pt.x - tlBound.x), floor(pt.y - tlBound.y)};
  }

  void renderTile(const size_t numGaussians, const Bounds2f& tileBounds, const unsigned workerId) {

    for (auto i = tileBounds.min.x ; i < tileBounds.max.x; ++i) {
      for (auto j = tileBounds.min.y  + workerId ; j < tileBounds.max.y; j+=numWorkers()) {

        float T = 1.0f;
        glm::vec4 colour = {0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec2 pixf = {(float) i, (float) j};

        for (auto gi = 0u; gi < numGaussians; ++gi) {
          Gaussian2D g = unpack<Gaussian2D>(gaus2D, gi * sizeof(Gaussian2D));

          glm::vec4 gCont = {g.colour.x, g.colour.y, g.colour.z, g.colour.w};
          // Conic is precomputed once at projection time — no per-pixel invert.
          const ivec3 conic = g.conic;
          const float opacity = g.colour.w;

          if (opacity == 0.0f) {
            continue;
          }
          ivec2 d = {g.mean.x - pixf.x, g.mean.y - pixf.y};
          float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
          if (power > 0.0f) {
            continue;
          }

          float alpha = ipu_min(0.99f, opacity * ipu_exp(power));
          if (alpha < 1.0f / 255.0f) {
            continue;
          }

          float test_T = T * (1.f - alpha);
          if (test_T < 0.0001f) {
              break;
          }

          colour += gCont * alpha * T;
          T = test_T;
        }


        // stop blending and apply colour to pixel
        ivec4 pixel = {colour.x, colour.y, colour.z, colour.w};
        auto pxTs = viewspaceToTile({pixf.x, pixf.y}, tileBounds.min);
        setPixel(pxTs.x, pxTs.y, pixel);
      }
    }
  }

  bool compute(unsigned workerId) {
    // Clear framebuffer — distributed across all workers
    unsigned word = 0;
    for (auto i = 4u * workerId; i < localFb.size(); i += 4u * numWorkers()) {
      memcpy(&localFb[i], &word, 4);
    }

    unsigned numToRender = splatted[0];
    if (numToRender > 0) {
      const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
      const auto tb = tfb.getTileBounds(tile_id[0]);
      renderTile(numToRender, tb, workerId);
    }

    return true;
  }

};


#define CCCSLOAD 80

// Template class to calculate register values
// for common compute state registers:
template <unsigned N, unsigned M>
struct CWEI {
  static constexpr unsigned value = M + (N * 4);
};

// This vertex loads the AMP engine weight matrix. This has to be done
// in supervisor mode. The weight registers are automatically zeroed on
// entering supervisor mode so we only need to load the non-zero parts.
class LoadMatrix : public poplar::SupervisorVertex {
public:
  // Specify the alignment and that the matrix must be in interleaved memory:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> matrix;

  bool compute() __attribute__((target("supervisor"))) {

    // Write the first address to load from into the $CCCSLOAD register:
    const auto loadStart = (unsigned)&matrix[0];

    // We want to load the 4x4 transform to upper left 4x4 block of the 16x16
    // common compute configuration registers $CWEI_N_M. Register indices are
    // calculated as index_of($CWEI_n_m) = m + n * 4.

    // Each ld128putcs instruction will read from the load address ($CCCSLOAD),
    // which must be in interleaved memory, and post increment it by 16 bytes:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_0_0 and CWEI_0_1:
    __builtin_ipu_ld128putcs(CWEI<0, 0>::value);
    // Load matrix slice [1, 0:3] to CWEI_1_0 and CWEI_1_1:
    __builtin_ipu_ld128putcs(CWEI<1, 0>::value);
    // Load matrix slice [2, 0:3] to CWEI_2_0 and CWEI_2_1:
    __builtin_ipu_ld128putcs(CWEI<2, 0>::value);
    // Load matrix slice [3, 0:3] to CWEI_3_0 and CWEI_3_1:
    __builtin_ipu_ld128putcs(CWEI<3, 0>::value);

    // Load the same 4x4 matrix into the lower right hand corner of weight matrix:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_4_2 and CWEI_4_3:
    __builtin_ipu_ld128putcs(CWEI<4, 2>::value);
    // Load matrix slice [1, 0:3] to CWEI_5_2 and CWEI_5_3:
    __builtin_ipu_ld128putcs(CWEI<5, 2>::value);
    // Load matrix slice [2, 0:3] to CWEI_6_2 and CWEI_6_3:
    __builtin_ipu_ld128putcs(CWEI<6, 2>::value);
    // Load matrix slice [3, 0:3] to CWEI_7_2 and CWEI_7_3:
    __builtin_ipu_ld128putcs(CWEI<7, 2>::value);

    return true;
  }
};

// Small piece of ASM required to zero the AMP accumulator registers:
inline
void zeroFpAccumulators() {
  asm(R"(
    setzi $a0, 0x8
    uput $FP_CLR, $a0
  )"
  :
  :
  : "$a0");
}

// This vertex uses the Accumulating Matrix Product (AMP) engine to transform 4x1 vectors by
// a single fixed 4x4 matrix. I.e. it is an optimised verison of the Transform4x4 vertex above.
//
// Accumulating Matrix Product (AMP) engine.
// =========================================
//
// A matrix-vector product can be interpreted as taking a linear combination of the columns of
// the matrix. I.e. a matrix projects a vector into its "column space": the vector space spanned
// by its columns. This is exactly how the AMP engine works: it is a "column scaling" engine (a
// systolic array) where partially scaled columns are fed to the next unit in the array and
// results accumulated until the results drop out of the end of the pipeline.
//
// Each amp instruction (f32sisoamp is used here, but there are different variants) takes scalar
// elements from the input vector one by one and feeds that scalar to every engine. Each engine
// then multiples the scalar with elements from the weight matrix and passes the intermediate
// result to the next engine which will add the contribution of the next column to it.
//
// Execution is organised into phases. Different phases connect different weights to different
// engines. These connections are made such that each engine in a phase is responsible for scaling
// a part of the column of the weight matrix and accumulating the result to the accumulators. So
// each phase scales and accumulates one column from the weight matrix. Once all phases are complete
// the results are ready, but can only be extracted from the pipeline two elements at a time (and
// only on even phases for f32sisoamp).
//
// Additionally the AMP instruction can take a partial result which is also added to the scaled
// column. This allows executing larger matrix multiples by decomposing them into smaller blocks:
// each block can load a partial result, add to it, and eventually save result back to memory (which
// can be reloaded again later and so on). In our use case here, we do not need partial inputs so
// they are always zero. This also enables us to clear the accumulators ready for the next iteration.
// However, this does mean that in this application the available FLOPS relating to partials are not
// utilised, so we can not expect to reach the peak FLOP rate of the machine where the calculation
// does not actively load partials.
class Transform4x4_amp : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vertsIn;
  poplar::Output<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vertsOut;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register const float* srcPtr asm("$m2") = &vertsIn[startIndex];
    register float* dstPtr asm("$m3") = &vertsOut[startIndex];
    const int span = vertsIn.size() - startIndex;
    unsigned iterations = span < 0 ? 0 : span / stride;

    asm (R"(
      .allow_optimizations

      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      {
        // Note we switch from using $a0:1 to using $a2:3 here to
        // free up more dual issue slots later:
        ld64step $a2:3, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      }
      {
        // Pre-load first input pair before entering the loop.
        // (Note we switch back to loads into $a0:1 ready for the loop):
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        // Optimised load/store instructions (ldst64pace below) require
        // triple packed addresses:
        tapack $m4:5, %[loadAddr], $mzero, %[storeAddr]
        f32sisoamp $azeros, $a2, $azeros, %[TAMP_F32_E4_P6]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
        .align 8
        {
          rpt %[iterations], 7
          f32sisoamp $azeros, $a3, $azeros, %[TAMP_F32_E4_P7] // This is not part of the loop
        }
        LOOP_START%=:
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        }
        {
          // Use stride specification to jump the packed read pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0001
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        }
        {
          // Use stride specification to jump the packed write pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0100 // At the end of the loop this is an over-read
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64pace $a2:3, $m4:5+=, $mzero, 0b00
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [iterations] "r"(iterations),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$m4:5", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};
