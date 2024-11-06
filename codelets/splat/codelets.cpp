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
#endif

ivec4 getTileColour(unsigned tid) {
  ivec4 c;
  if (tid % 3 == 0) {
    c = {.1f, 0.0f, 0.0f, 0.0f};
  } else if (tid % 3 == 1) {
    c = {0.0f, .1f, 0.0f, 0.0f};
  } else {
    c = {0.0f, 0.0f, .1f, 0.0f};
  }
  return c;
}

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

class CullGaussians : public poplar::MultiVertex {
public:

  poplar::Input<poplar::Vector<float>> vertsIn;
  poplar::Output<poplar::Vector<unsigned>> depths;

  poplar::Input<poplar::Vector<float>> modelView;
  poplar::Input<poplar::Vector<float>> projection;

  poplar::Input<poplar::Vector<int>> tile_id;

  void cullInternal(const glm::mat4& viewmatrix, const TiledFramebuffer& tfb, const splat::Viewport& vp) {
    // float max = -1.0f;
    for (auto i = 0; i < vertsIn.size(); i+=sizeof(Gaussian3D)) {
      auto idx = i / sizeof(Gaussian3D);
      Gaussian3D g = unpack<Gaussian3D>(vertsIn, i);
      depths[idx] <<= 16;

      if (g.gid <= 0) {
        continue;
      }

      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, g.mean.w};
      auto clipSpace = viewmatrix * glmMean;

      // perform near plane frustum culling
      if (clipSpace.z > 0.f) {
        continue;
      }

      // write the depth value to the lower bits of tid float value
      // auto z = half(-clipSpace.z);
      // unsigned key;
      // memcpy(&key, &z, sizeof(z));
      // key >>= 16;
      // key |= depths[idx];

      auto z = -clipSpace.z;
      depths[idx] |= *(unsigned*)&z;
      // depths[idx] = key;
    }
  }

  bool compute(unsigned workerId) {

    // construct mapping from tile to framebuffer
    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    // Transpose because GLM storage order is column major:
    const auto viewmatrix = glm::transpose(glm::make_mat4(&modelView[0]));
    const auto projmatrix = glm::transpose(glm::make_mat4(&projection[0]));
    const auto mvp = projmatrix * viewmatrix;
    cullInternal(mvp, tfb, vp);

    return true;
  }

};

// Multi-Vertex to transform every 4x1 vector
// in an array by the same 4x4 transformation matrix.
// Uses the OpenGL Math (GLM) library for demonstration
// purposes (we do not expect this to be fast as GLM is
// not optimised for IPU yet).
//
// This is here as a reference to show what the
// accumulating matrix product (AMP) engine assembly
// vertices below are doing.

class GSplat : public poplar::MultiVertex {

public:
  poplar::Input<poplar::Vector<float>> modelView;
  poplar::Input<poplar::Vector<float>> projection;

  poplar::Input<poplar::Vector<int>> tile_id;
  poplar::Input<poplar::Vector<float>> fxy;
  poplar::Output<poplar::Vector<unsigned>> splatted;
  
  poplar::InOut<poplar::Vector<float>> vertsIn;
  poplar::Output<poplar::Vector<int>> indices;
  poplar::Output<poplar::Vector<float>> gaus2D;

  poplar::Output<poplar::Vector<float>> localFb;

  poplar::Input<poplar::Vector<float>> rightIn;
  poplar::Output<poplar::Vector<float>> rightOut;

  poplar::Input<poplar::Vector<float>> leftIn; 
  poplar::Output<poplar::Vector<float>> leftOut;

  poplar::Input<poplar::Vector<float>> upIn;
  poplar::Output<poplar::Vector<float>> upOut;

  poplar::Input<poplar::Vector<float>> downIn;
  poplar::Output<poplar::Vector<float>> downOut;

  float clipSize;


  unsigned toByteBufferIndex(float x, float y) {
    return unsigned(x + y * IPU_TILEWIDTH) * 4;
  } 

  void setPixel(float x, float y, const ivec4 &colour) {
    ivec4 pixel;
    unsigned idx = toByteBufferIndex(x, y);
    memcpy(&pixel, &localFb[idx], sizeof(pixel));
    // if (pixel.w >= 100.f) {
    //   return;
    // }

    pixel = pixel + colour;
    memcpy(&localFb[idx], &pixel, sizeof(pixel));
  }

  ivec2 viewspaceToTile(const ivec2& pt, ivec2 tlBound) {
    return {floor(pt.x - tlBound.x), floor(pt.y - tlBound.y)};
  }

  template<typename G> bool send(const G &g, directions dirs) {
    // if no dirs set then we still return true.. returning true means 
    // the gaussian is held somewhere. Either this tile or in an out buf. in this case
    // it will remain in vertsIn since send is never called with evict
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

  void colourFb(const ivec4 &colour, unsigned workerId) {
    const auto startIndex = 4 * workerId;
    for (auto i = startIndex; i < localFb.size(); i += 4 * numWorkers()) {
      memcpy(&localFb[i], &colour, sizeof(colour));
    }
  }

  void addBG(const ivec4 &colour) {
    for (auto i = 0; i < localFb.size(); i += 4) {
      ivec4 pixel;
      memcpy(&pixel, &localFb[i], sizeof(pixel));
      pixel = pixel + colour;
      memcpy(&localFb[i], &pixel, sizeof(pixel));
    }
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
    float pivot = pivotG.z;
    int i = low - sizeof(G);
    for (int j = low; j <= high - sizeof(G); j+=sizeof(G)) {
        G gm;
        std::memcpy(&gm, &elements[j], sizeof(G));
        if (gm.z <= pivotG.z) {
            i+=sizeof(G);
            swap<G>(&elements[i], &elements[j]);
        }
    }
    swap<G>(&elements[i + sizeof(G)], &elements[high]);
    return (i + sizeof(G)) / sizeof(G);
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

  template<typename G>
  void sortBuffer(poplar::Vector<float>& buffer, unsigned end, unsigned workerId = 0u) {
    if (end < 1 || end >= indices.size() - 1) {
      return;
    }
    // zero the indices
    for (auto i = 0u; i < indices.size(); ++i) {
      indices[i] = 0;
    }
    iterativeQuickSort<G>(&buffer[0], 0, end);
  }

  void renderTile(const size_t numGaussians, const Bounds2f& tileBounds, const unsigned workerId = 0u) {

    sortBuffer<Gaussian2D>(gaus2D, numGaussians);

    // const auto startIndex = sizeof(Gaussian3D) * workerId;
    // for (auto i = startIndex; i < rightOut.size(); i+=sizeof(Gaussian3D) * numWorkers()) {

    for (auto i = tileBounds.min.x ; i < tileBounds.max.x; ++i) {
      for (auto j = tileBounds.min.y  + workerId ; j < tileBounds.max.y; j+=numWorkers()) {

        float T = 1.0f;
        glm::vec4 colour = {0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec2 pixf = {(float) i, (float) j};

        for (auto gi = 0u; gi < numGaussians; ++gi) {
          Gaussian2D g = unpack<Gaussian2D>(gaus2D, gi * sizeof(Gaussian2D));

          glm::vec4 gCont = {g.colour.x, g.colour.y, g.colour.z, g.colour.w};
          ivec4 con_o = g.ComputeConicOpacity();

          if (con_o.w == 0.0) {
            continue;
          }
          ivec2 xy = g.mean;
          ivec2 d = {xy.x - pixf.x, xy.y - pixf.y};
          float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
          if (power > 0.0f) {
            continue;
          }

          float alpha = glm::min(0.99f, con_o.w * exp(power));
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

  unsigned rasterise(const Gaussian2D &g, const Bounds2f& bb, const Bounds2f& tb) {
    auto count = 0u;
    auto centre = glm::vec2(g.mean.x, g.mean.y);
    for (auto i = bb.min.x; i < bb.max.x; i++) {
      for (auto j = bb.min.y; j < bb.max.y; j++) {
        auto px = viewspaceToTile({i, j}, tb.min);
        if(g.inside(i,j)) {
          setPixel(px.x, px.y, g.colour);
        } 
      }
    }
    return count;
  }

  template<typename InternalStorage> void renderInternal(InternalStorage& buffer,
                                                         const glm::mat4& projmatrix,
                                                         const glm::mat4& viewmatrix,
                                                         const TiledFramebuffer& tfb, 
                                                         const splat::Viewport& vp,
                                                         const unsigned workerId = 0u) {
    const auto tb = tfb.getTileBounds(tile_id[0]);
    const auto mvp = projmatrix * viewmatrix;
    glm::vec2 tanfov(tan(0.5 * fxy[0]),
                      tan(0.5 * fxy[0]));

      glm::vec2 focal(tfb.width / (2.f * glm::tan(fxy[0] / 2.f)),
            tfb.height / (2.f * glm::tan(fxy[0] / 2.f)));


    auto toRender = 0u;
    for (auto i = 0; i < buffer.size(); i+=sizeof(Gaussian3D)) {

      Gaussian3D g = unpack<Gaussian3D>(buffer, i);
      auto scale = g.scale;
      if (g.gid <= 0) {
        continue;
      }

      auto clipSpace = mvp * glm::vec4(g.mean.x, g.mean.y, g.mean.z, g.mean.w);
      auto projMean = vp.clipSpaceToViewport(clipSpace);

      g.scale = g.scale / fxy[1];
      // render and clip, send to the halo region around the current tile
      ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tanfov.x, tanfov.y, focal.x, focal.y);
      Gaussian2D g2D({projMean.x, projMean.y}, g.colour, cov2D, clipSpace.z);
      auto bb = g2D.GetBoundingBox();
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

      if (withinGuardBand && g2D.z < 0.0f ) {
        auto g2Idx = toRender * sizeof(Gaussian2D);
        insertAt(gaus2D, g2Idx, g2D);
        toRender++;
        // rasterise(g2D, bb, tb);
      }
    }

    if (toRender > 0) {
      renderTile(toRender, tb, workerId);
      splatted[0] = toRender;
    }
  }

  void readInput(poplar::Input<poplar::Vector<float>> &bufferIn,
                                    const direction& recievedFrom,
                                    const glm::mat4& projmatrix,
                                    const glm::mat4& viewmatrix,
                                    const TiledFramebuffer& tfb,
                                    const splat::Viewport& vp) {
    // Get the boundary of the current tile's framebuffer section
    const auto tb = tfb.getTileBounds(tile_id[0]);
    const auto tbPrev = tfb.getTileBounds(tfb.getNearbyTile(tile_id[0], recievedFrom));

    glm::vec2 tanfov(glm::tan(0.5 * fxy[0]),
                    glm::tan(0.5 * fxy[0]));

    glm::vec2 focal(tfb.width / (2.f * glm::tan(fxy[0] / 2.f)),
                    tfb.height / (2.f * glm::tan(fxy[0] / 2.f)));

    const auto mvp = projmatrix * viewmatrix;

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

      // project the 3D gaussian into 2D using EWA splatting algorithm
      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, g.mean.w};
      auto clipSpace = mvp * glmMean;
      auto projMean = vp.clipSpaceToViewport(clipSpace);
   
      if (tb.contains(ivec2{projMean.x, projMean.y})) {
        // anchor arrived so we insert and let
        // render main handle the rest
        bool overflow = !insert(vertsIn, g);
        continue;
      } 

      g.scale = g.scale / fxy[1];

      ivec3 cov2D = g.ComputeCov2D(projmatrix, viewmatrix, tanfov.x, tanfov.y, focal.x, focal.y);
      Gaussian2D g2D({projMean.x, projMean.y}, g.colour, cov2D, clipSpace.z);
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

      // the gaussian is being propagated away from the anchor,
      // we need to render and pass it on until the extent is fully rendered.
      auto bb = g2D.GetBoundingBox();

      if (bb.diagonal().length() < tb.diagonal().length() * clipSize) {
        directions sendTo;
        auto clippedBB = bb.clip(tb, sendTo);
        protocol<Gaussian3D>(g, sendTo, recievedFrom);
      }
      bool overflow = !insert(vertsIn, g);

    }
  } 

  void clearOutBuffers(unsigned workerId) {
    const auto startIndex = sizeof(Gaussian3D) * workerId;
    for (auto i = startIndex; i < rightOut.size(); i+=sizeof(Gaussian3D) * numWorkers()) {
      evict<Gaussian3D>(rightOut, i);
    }
    for (auto i = startIndex; i < leftOut.size(); i+=sizeof(Gaussian3D) * numWorkers()) {
      evict<Gaussian3D>(leftOut, i);
    }
    for (auto i = startIndex; i < upOut.size(); i+=sizeof(Gaussian3D) * numWorkers()) {
      evict<Gaussian3D>(upOut, i);
    }
    for (auto i = startIndex; i < downOut.size(); i+=sizeof(Gaussian3D) * numWorkers()) {
      evict<Gaussian3D>(downOut, i);
    }
  }


  bool compute(unsigned workerId) {

    // zero the framebuffer 
    auto black = ivec4{0.0f, 0.0f, 0.0f, 0.0f};
    // getTileColour(tile_id[0])
    colourFb(black, workerId);

     //clear all of the out buffers:
    clearOutBuffers(workerId);

    // construct mapping from tile to framebuffer
    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    clipSize =  12.0f;

    // Transpose because GLM storage order is column major:
    const auto viewmatrix = glm::transpose(glm::make_mat4(&modelView[0]));
    const auto projmatrix = glm::transpose(glm::make_mat4(&projection[0]));

    readInput(rightIn, direction::right, projmatrix, viewmatrix, tfb, vp);
    readInput(leftIn, direction::left, projmatrix, viewmatrix, tfb, vp);
    readInput(upIn, direction::up, projmatrix, viewmatrix, tfb, vp);
    readInput(downIn, direction::down, projmatrix, viewmatrix, tfb, vp);

    renderInternal(vertsIn, projmatrix, viewmatrix, tfb, vp, workerId);
    

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
