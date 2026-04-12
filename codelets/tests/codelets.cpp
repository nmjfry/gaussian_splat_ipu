// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/tileMapping/tile_config.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#define CHECK_EQUAL(a,b) \
do { \
  if ((a) != (b)) { \
    printf("\n"); assert(a == b); \
  } \
} while(0)

#define CHECK_LE(a,b) \
do { \
  if ((a) > (b)) { \
    printf("\n"); assert(a > b); \
  } \
} while(0)

// Test that GLM works on IPU:
class GlmMat4 : public poplar::Vertex {
public:
  bool compute() {
    float m[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16};
    float v[] = {2, 4, 6, 8};
    const auto mgl = glm::transpose(glm::make_mat4(m));
    auto vgl = glm::make_vec4(v);
    vgl = mgl * vgl;
    CHECK_EQUAL(vgl.x, 60);
    CHECK_EQUAL(vgl.y, 140);
    CHECK_EQUAL(vgl.z, 220);
    CHECK_EQUAL(vgl.w, 300);
    return true;
  }
};

class GlmTransform : public poplar::Vertex {
public:
  bool compute() {
    glm::vec4 position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(10.f, 10.f, 10.f),
                                 glm::vec3(0.f, 0.f, 0.0),
                                 glm::vec3(0.f, 1.f, 0.0));

    glm::vec4 transformed = view * position;
    CHECK_EQUAL(transformed.x, 0.f);
    CHECK_EQUAL(transformed.y, 0.f);
    CHECK_EQUAL(transformed.w, 1.f);
    auto zAbsErr = abs(-sqrt(300.f) - transformed.z );
    CHECK_LE(zAbsErr, 0.00001f);
    return true;
  }
};

class TfbBoundsCheck : public poplar::Vertex {
public:
  bool compute() {
    const splat::TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);

    auto tb = tfb.getTileBounds(3);
    auto tb1 = tfb.getTileBounds(1);
    auto dist = tfb.manhattanDistance(tb.min, tb1.min);

    CHECK_EQUAL(dist, 2 * IPU_TILEWIDTH);

    auto mid120 = tfb.getTileBounds(120).centroid();
    auto next = tfb.getNearbyTile(120, splat::direction::right);
    auto down = tfb.getNearbyTile(120, splat::direction::down);

    auto tb2 = tfb.getTileBounds(next);
    auto tb3 = tfb.getTileBounds(down);

    auto dist2 = tfb.manhattanDistance(mid120, tb2.centroid());
    auto dist3 = tfb.manhattanDistance(mid120, tb3.centroid());

    CHECK_EQUAL(dist2, IPU_TILEWIDTH);
    CHECK_EQUAL(dist3, IPU_TILEHEIGHT);

    return true;
  }
};

class GaussianTests : public poplar::Vertex {
  public:
  bool compute() {
    const splat::TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);

    splat::Gaussian3D g;
    g.colour = {.4f, 0.f, 0.f, 0.9f};
    g.mean = {0.f, 0.f, 0.f};
    g.gid = 9;
    g.scale = {1.0f, 1.0f, 1.0f};

    splat::Gaussian2D g2D({640.f, 360.f}, g.colour, {4.f, 0.5f, 4.f}, 0.f);

    const auto tb = tfb.getTileBounds(0);
    const auto tb2 = tfb.getTileBounds(40);
    const auto tb3 = tfb.getTileBounds(39);

    auto dstTile = tfb.pixCoordToTile(g2D.mean.x, g2D.mean.y);
    printf("dstTile: %f\n", dstTile);
    auto dstCentre = tfb.getTileBounds(dstTile).centroid();
    printf("src centre: %f %f\n", tb.centroid().x, tb.centroid().y);
    auto direction = tfb.getBestDirection(tb.centroid(), dstCentre);

    CHECK_EQUAL(direction, splat::direction::right);

    auto dir = tfb.getBestDirection(tb.centroid(), tb2.centroid());
    CHECK_EQUAL(dir, splat::direction::down);

    auto dir2 = tfb.getBestDirection(tb2.centroid(), tb3.centroid());
    CHECK_EQUAL(dir2, splat::direction::right);

    auto dir3 = tfb.getBestDirection(tb2.centroid(), tb.centroid());
    CHECK_EQUAL(dir3, splat::direction::up);

    auto dir4 = tfb.getBestDirection(tb2.centroid(), tb2.centroid());
    CHECK_EQUAL(dir4, splat::direction::none);

    auto dir5 = tfb.getBestDirection(tb3.centroid(), tb.centroid());
    CHECK_EQUAL(dir5, splat::direction::left);


    return true;
  }
};
