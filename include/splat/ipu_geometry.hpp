
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/math/sincos.hpp>

#ifdef __IPU__
#include <ipu_builtins.h>
inline float geo_exp(float x) { return __builtin_ipu_exp(x); }
inline float geo_max(float a, float b) { return __builtin_ipu_max(a, b); }
inline float geo_min(float a, float b) { return __builtin_ipu_min(a, b); }
#else
inline float geo_exp(float x) { return expf(x); }
inline float geo_max(float a, float b) { return a > b ? a : b; }
inline float geo_min(float a, float b) { return a < b ? a : b; }
#endif

// #ifdef __IPU__
// #else 
//   #include <sincos.hpp>
// #endif

namespace splat {

struct ivec4 {
  float x;
  float y;
  float z;
  float w;
  struct ivec4 operator+(ivec4 const &other) const {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
  }
  struct ivec4 operator-(ivec4 const &other) const {
    return {x - other.x, y - other.y, z - other.z, w - other.w};
  }
  struct ivec4 operator*(float &scalar) const {
    return {x * scalar, y * scalar, z * scalar, w * scalar};
  }
  bool operator==(ivec4 const &other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }
};

typedef struct ivec4 ivec4;

struct ivec2 {
  float x;
  float y;
  struct ivec2 operator+(ivec2 const &other) const {
    return {x + other.x, y + other.y};
  }
  struct ivec2 operator-(ivec2 const &other) const {
    return {x - other.x, y - other.y};
  }
  struct ivec2 operator*(float const &scalar) const {
    return {x * scalar, y * scalar};
  }
  struct ivec2 operator/(float const &scalar) const {
    return {x / scalar, y / scalar};
  }
  float length() const {
    return sqrt(x * x + y * y);
  }
};

typedef struct ivec2 ivec2;

struct ivec3 {
  float x;
  float y;
  float z;
  struct ivec3 operator+(ivec3 const &other) {
    return {x + other.x, y + other.y, z + other.z};
  }
  struct ivec3 operator-(ivec3 const &other) {
    return {x - other.x, y - other.y, z - other.z};
  }
  struct ivec3 operator*(float const &scalar) {
    return {x * scalar, y * scalar, z * scalar};
  }
  struct ivec3 operator/(float const &scalar) {
    return {x / scalar, y / scalar, z / scalar};
  }
};

typedef struct ivec3 ivec3;

typedef struct directions {
    bool up;
    bool right;
    bool down;
    bool left;
    bool keep;
    static const int NUM_DIRS = 4;
    bool any() const {
      return up || right || down || left;
    }
    bool none() const {
      return !any();
    }
} directions;

enum direction {
  left,
  right,
  up,
  down,
  none
};

struct Bounds2f {
  Bounds2f(bool) {
    // Overload to skip default init. Used to preseve contents on references.
  }

  Bounds2f(const ivec2& _min, const ivec2& _max) : min(_min), max(_max) {}

  ivec2 centroid() const {
    return (max + min) * .5f;
  }

  ivec2 diagonal() const {
    return max - min;
  }

  // /// Extend the bounds to enclose another bounding box:
  // void operator += (const Bounds2f& other) {
  //   min.x = min(min.x, other.min.x);
  //   min.y = min(min.y, other.min.y);
  //   max.x = max(max.x, other.max.x);
  //   max.y = max(max.y, other.max.y);
  // }

  // /// Extend the bounds to enclose the specified point:
  // void operator += (const ivec2& v) {
  //   min.x = min(min.x, v.x);
  //   min.y = min(min.y, v.y);
  //   max.x = max(max.x, v.x);
  //   max.y = max(max.y, v.y);
  // }

  Bounds2f clip(const Bounds2f& fixedBound, directions& dirs) const {
    ivec2 topleft = min;
    ivec2 bottomright = max;
    dirs.left = floor(topleft.x) < fixedBound.min.x;
    dirs.up = floor(topleft.y) < fixedBound.min.y;
    dirs.right = ceil(bottomright.x) >= fixedBound.max.x;
    dirs.down = ceil(bottomright.y) >= fixedBound.max.y;

    if (dirs.left) {
      topleft.x = fixedBound.min.x;
    }
    if (dirs.up) {
      topleft.y = fixedBound.min.y;
    }
    if (dirs.right) {
      bottomright.x = fixedBound.max.x;
    }
    if (dirs.down) {
      bottomright.y = fixedBound.max.y;
    }
    Bounds2f clipped = {topleft, bottomright};
    return clipped;
  }


  Bounds2f clip(const Bounds2f& fixedBound) const {
    directions dirs;
    return clip(fixedBound, dirs);
  }

  bool contains(const ivec2& v) const {
    return ceil(v.x) >= min.x && floor(v.x) < max.x && ceil(v.y) >= min.y && floor(v.y) < max.y;
  }

  bool contains(const ivec4& v) const {
    return contains(ivec2{v.x, v.y});
  }

  void print() const {
    printf("tl : %f, %f  br: %f, %f", min.x, min.y, max.x, max.y);
  }

  ivec2 min;
  ivec2 max;
};

struct Primitive {
  ivec4 mean; // in world space
  ivec4 colour; // RGBA colour space
  float gid;
  virtual Bounds2f getBoundingBox() const = 0;  
  virtual bool inside(float x, float y) const = 0;
};

struct square : Primitive {
  float radius = 10.f;

  Bounds2f getBoundingBox() const override {
    return Bounds2f({mean.x - radius, mean.y - radius}, {mean.x + radius, mean.y + radius});
  }

  bool inside(float x, float y) const override {
    return true;
  }

  static bool isOnTile(ivec2 pos, ivec2 tlBound, ivec2 brBound) {
    return pos.x >= tlBound.x && pos.x <= brBound.x && pos.y >= tlBound.y && pos.y <= brBound.y;
  }

  static directions clip(ivec2 tlBound, ivec2 brBound, ivec2& topleft, ivec2& bottomright) {
    directions dirs;


    dirs.left = topleft.x < tlBound.x;
    dirs.up = topleft.y < tlBound.y;
    dirs.right = bottomright.x >= brBound.x;
    dirs.down = bottomright.y >= brBound.y;

    ivec2 topright = {bottomright.x, topleft.y};
    ivec2 bottomleft = {topleft.x, bottomright.y};
    dirs.keep = isOnTile(topleft, tlBound, brBound) || isOnTile(bottomright, tlBound, brBound) 
    || isOnTile(topright, tlBound, brBound) || isOnTile(bottomleft, tlBound, brBound);

    if (dirs.left) {
      topleft.x = tlBound.x;
    }
    if (dirs.up) {
      topleft.y = tlBound.y;
    }
    if (dirs.right) {
      bottomright.x = brBound.x;
    }
    if (dirs.down) {
      bottomright.y = brBound.y;
    }
    return dirs;
  }
};

struct Gaussian2D {
  ivec4 colour;  // RGB + opacity in .w
  ivec3 conic;   // precomputed inverse of 2D covariance
  ivec2 mean;    // screen space
  float z;

  Gaussian2D() {}

  // Construct from 2D covariance — compute conic once here so the pixel loop
  // doesn't have to re-invert a 2x2 for every pixel × every Gaussian.
  Gaussian2D(ivec2 _mean, ivec4 _colour, ivec3 _cov2D, float _z)
    : colour(_colour), mean(_mean), z(_z) {
    float det = _cov2D.x * _cov2D.z - _cov2D.y * _cov2D.y;
    if (det == 0.0f) {
      conic = {0.f, 0.f, 0.f};
    } else {
      float det_inv = 1.f / det;
      conic = { _cov2D.z * det_inv, -_cov2D.y * det_inv, _cov2D.x * det_inv };
    }
  }

  // Bounding-radius method from the original 3DGS: 3 standard deviations of the
  // larger eigenvalue. Static so it can be computed before constructing Gaussian2D
  // (since we only store the conic now, not the covariance).
  static Bounds2f BoundingBoxFromCov(const ivec2& mean, const ivec3& cov2D) {
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    float mid = .5f * (cov2D.x + cov2D.z);
    float lambda1 = mid + glm::sqrt(geo_max(0.1f, mid * mid - det));
    float lambda2 = mid - glm::sqrt(geo_max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(geo_max(lambda1, lambda2)));
    return Bounds2f({mean.x - my_radius, mean.y - my_radius},
                    {mean.x + my_radius, mean.y + my_radius});
  }
};

class Gaussian3D {
  public:
    // Packed layout: 60 bytes total (was 64 — dropped mean.w which was always 1.0).
    // gid must remain the last element: insert()/evict() helpers locate it by
    // offset (sizeof(g) - sizeof(float)).
    ivec3 mean;   // world space, w was always 1.0
    ivec4 colour; // RGB + opacity
    ivec4 rot;    // quaternion (real, i, j, k)
    ivec3 scale;  // log-space per-axis
    float gid;

    // Convert from (scale, rot) into the gaussian covariance matrix in world space.
    // Matches original 3DGS (forward.cu): M = S * R, Sigma = M^T * M
    glm::mat3 ComputeCov3D() const
    {
        // Quaternion: rot = (w, x, y, z) per 3DGS .ply convention
        float r = rot.x, x = rot.y, y = rot.z, z = rot.w;

        // Build rotation matrix exactly as the original CUDA code does
        // (GLM column-major: each argument triple is a column)
        glm::mat3 R = glm::mat3(
            1.f - 2.f*(y*y + z*z), 2.f*(x*y - r*z),       2.f*(x*z + r*y),
            2.f*(x*y + r*z),       1.f - 2.f*(x*x + z*z),  2.f*(y*z - r*x),
            2.f*(x*z - r*y),       2.f*(y*z + r*x),         1.f - 2.f*(x*x + y*y)
        );

        glm::mat3 S(1.0f);
        S[0][0] = geo_exp(scale.x);
        S[1][1] = geo_exp(scale.y);
        S[2][2] = geo_exp(scale.z);

        glm::mat3 M = S * R;
        return glm::transpose(M) * M;
    }

    ivec3 ComputeCov2D(const glm::mat4& projmatrix, const glm::mat4& viewmatrix, float tan_fovx, float tan_fovy, float focal_x, float focal_y) {
      glm::vec3 t = glm::vec3(viewmatrix * glm::vec4(mean.x, mean.y, mean.z, 1.0f));
      const float limx = 1.3f * tan_fovx;
      const float limy = 1.3f * tan_fovy;
      const float txtz = t.x / t.z;
      const float tytz = t.y / t.z;
      t.x = geo_min(limx, geo_max(-limx, txtz)) * t.z;
      t.y = geo_min(limy, geo_max(-limy, tytz)) * t.z;

      glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

      // W extraction must match the original CUDA code which reads from
      // a row-major float[16]. In GLM's column-major layout this is
      // equivalent to transposing the upper-left 3x3:
      glm::mat3 W = glm::transpose(glm::mat3(viewmatrix));
      glm::mat3 T = W * J;

      glm::mat3 cov3D = ComputeCov3D();

      // Original 3DGS: cov = T^T * transpose(Vrk) * T
      // Vrk is symmetric so transpose(Vrk) == Vrk.
      glm::mat3 cov = glm::transpose(T) * glm::transpose(cov3D) * T;

      // Apply low-pass filter: every Gaussian should be at least
      // one pixel wide/high. Discard 3rd row and column.
      cov[0][0] += 0.3f;
      cov[1][1] += 0.3f;

      return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
    }
};

#define GAUSSIAN_SIZE sizeof(Gaussian3D)

} 