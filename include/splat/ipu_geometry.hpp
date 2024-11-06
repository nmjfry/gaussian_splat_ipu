
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/math/sincos.hpp>

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
  ivec4 colour; // RGBA colour space
  ivec3 cov2D;
  ivec2 mean; // in screen space 
  float z;

  Gaussian2D(ivec2 _mean, ivec4 _colour, ivec3 _cov2D, float z) : mean(_mean), colour(_colour), cov2D(_cov2D), z(z) {}

  Gaussian2D() {}

  static float max(float a, float b) {
    return a > b ? a : b;
  }

  // computes the eigenvalues and rotation from the 2D covariance matrix
  ivec3 ComputeEigenvalues() const {
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    float mid = .5f * (cov2D.x + cov2D.z);
    float lambda1 = mid + glm::sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - glm::sqrt(max(0.1f, mid * mid - det));
    float theta;
    if (cov2D.y == 0 && cov2D.x >= cov2D.z) {
      theta = 0;
    } else if (cov2D.y == 0 && cov2D.x < cov2D.z) {
      theta = glm::pi<float>() / 2.f;
    } else {
      theta = glm::atan(lambda1 - cov2D.x, cov2D.y);
    }
    return {lambda1, lambda2, theta};
  }

  Bounds2f GetBoundingBox() const {
    auto [e1, e2, theta] = ComputeEigenvalues();
    float c, s;
    sincos(theta, s, c);
    auto dd = (e1 / 2) * (e1 / 2);
    auto DD = (e2 / 2) * (e2 / 2);
    auto dxMax = glm::sqrt(dd * (c * c) + DD * (s * s));
    auto dyMax = glm::sqrt(dd * (s * s) + DD * (c * c));

    float my_radius = ceil(3.f * sqrt(max(e1, e2)));
    // return Bounds2f({mean.x - dxMax, mean.y - dyMax}, {mean.x + dxMax, mean.y + dyMax});

    return Bounds2f({mean.x - my_radius, mean.y - my_radius}, {mean.x + my_radius, mean.y + my_radius});
  }

  ivec4 ComputeConicOpacity() const {
    // Invert covariance (EWA algorithm)
    float det = (cov2D.x * cov2D.z - cov2D.y * cov2D.y);
    if (det == 0.0f)
      return {0.f, 0.f, 0.f, 0.f};
    float det_inv = 1.f / det;
    ivec4 conic = { cov2D.z * det_inv, -cov2D.y * det_inv, cov2D.x * det_inv, colour.w };
    return conic;
  }

  // Pixel test to see if a pixel is inside the gaussian
  bool inside(float x, float y) const {
    auto es = ComputeEigenvalues();
    auto theta = es.z;
    float c, s;
    sincos(theta, s, c);
    auto e1 = es.x;
    auto e2 = es.y;
    auto dd = (e1 / 2) * (e1 / 2);
    auto DD = (e2 / 2) * (e2 / 2);
    auto a = c * (x - mean.x) + s * (y - mean.y);
    auto b = s * (x - mean.x) - c * (y - mean.y);
    return (((a * a) / dd)  + ((b * b) / DD)) <= 1;
  }

};

class Gaussian3D {
  public:
    ivec4 mean; // in world space
    ivec4 colour; // RGBA colour space
    ivec4 rot;  // local rotation of gaussian (real, i, j, k)
    ivec3 scale;
    float gid;

    // convert from (scale, rot) into the gaussian covariance matrix in world space
    // See 3d Gaussian Splat paper for more info
    glm::mat3 ComputeCov3D() const
    {
        glm::quat q(rot.x, rot.y, rot.z, rot.w);
        glm::mat3 R(glm::normalize(q));
        glm::mat3 S(glm::vec3(expf(scale.x), 0.0f, 0.0f),
                    glm::vec3(0.0f, expf(scale.y), 0.0f),
                    glm::vec3(0.0f, 0.0f, expf(scale.z)));
        return  R *  S * glm::transpose(S) * glm::transpose(R);
    }

    static float max(float a, float b) {
      return a > b ? a : b;
    }

    static float min(float a, float b) {
      return a < b ? a : b;
    }

    ivec3 ComputeCov2D(const glm::mat4& projmatrix, const glm::mat4& viewmatrix, float tan_fovx, float tan_fovy, float focal_x, float focal_y) {
      // t is in view coordinates
      // float alpha = mean.w;
      // glm::vec4 t = viewmatrix * glm::vec4(mean.x, mean.y, mean.z, 1.0f);

      const glm::mat4 mv = projmatrix * viewmatrix;
      glm::vec3 t = glm::vec3(mv * glm::vec4(mean.x, mean.y, mean.z, 1.0f));
      const float limx = 1.3f * tan_fovx;
      const float limy = 1.3f * tan_fovy;
      const float txtz = t.x / t.z;
      const float tytz = t.y / t.z;
      t.x = min(limx, max(-limx, txtz)) * t.z;
      t.y = min(limy, max(-limy, tytz)) * t.z;

      glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

      // glm::mat3 W = glm::mat3(mv);

      glm::mat3 T = glm::mat3(mv) * J;

      glm::mat3 cov3D = ComputeCov3D();

      glm::mat3 cov = T * cov3D * glm::transpose(T);

      // Apply low-pass filter: every Gaussian should be at least
      // one pixel wide/high. Discard 3rd row and column.
      cov[0][0] += 0.3f;
      cov[1][1] += 0.3f;

      return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
    }
};

#define GAUSSIAN_SIZE sizeof(Gaussian3D)

} 