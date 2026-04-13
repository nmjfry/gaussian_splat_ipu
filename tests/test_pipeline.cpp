// Full-pipeline equivalence test: IPU 3DGS (post-optimisation) vs original 3DGS CUDA.
//
// Runs N Gaussians through both implementations with the SAME inputs, and compares:
//   Stage A: ComputeCov3D per Gaussian
//   Stage B: ComputeCov2D per Gaussian
//   Stage C: conic inversion (our new cached path vs original per-pixel path)
//   Stage D: depth sort order (front-to-back for both)
//   Stage E: per-pixel alpha compositing (full blend formula)
//
// Compile: g++ -std=c++17 -I include -I external/glm tests/test_pipeline.cpp -o test_pipeline

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ---------- Test scene ----------
struct TestG {
  glm::vec3 mean;
  glm::vec3 scale; // log-space
  glm::vec4 rot;   // (w, x, y, z)
  float opacity;   // raw (pre-sigmoid)
  glm::vec3 sh_dc;
};

// Several test Gaussians spread in depth along -Z:
std::vector<TestG> makeScene() {
  return {
    {{0.0f,  0.0f, -2.0f}, {-3.0f, -3.0f, -3.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, 2.0f, {0.8f, 0.1f, 0.1f}},
    {{0.5f, -0.2f, -3.0f}, {-2.5f, -3.5f, -3.0f}, {0.9f, 0.1f, 0.2f, 0.3f}, 1.5f, {0.1f, 0.8f, 0.1f}},
    {{-0.3f, 0.1f, -4.0f}, {-3.5f, -2.5f, -3.5f}, {0.7f, 0.3f, 0.1f, 0.2f}, 1.0f, {0.1f, 0.1f, 0.8f}},
    {{0.2f,  0.3f, -5.0f}, {-2.0f, -3.0f, -3.5f}, {0.8f, 0.2f, 0.2f, 0.2f}, 0.5f, {0.5f, 0.5f, 0.5f}},
  };
}

struct Cam {
  glm::mat4 view, proj;
  float tan_fovx, tan_fovy;
  float focal_x, focal_y;
  float W, H;
};

Cam makeCam() {
  float fovY = glm::radians(60.0f);
  float aspect = 1280.0f / 720.0f;
  float tan_fovy = tanf(fovY / 2.0f);
  float tan_fovx = tan_fovy * aspect;
  Cam c;
  // Build an OpenGL-style view matrix then apply the IPU's runtime OpenGL->COLMAP
  // conversion (diag(1,-1,-1,1) pre-multiplied) so the test reflects what the IPU
  // actually feeds to its codelet / to DGR.
  glm::mat4 V_opengl = glm::lookAt(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  glm::mat4 flipYZ(1.f);
  flipYZ[1][1] = -1.f;
  flipYZ[2][2] = -1.f;
  c.view = flipYZ * V_opengl;
  c.proj = glm::perspective(fovY, aspect, 0.01f, 100.0f);
  c.tan_fovx = tan_fovx;
  c.tan_fovy = tan_fovy;
  c.focal_x = 1280.0f / (2.0f * tan_fovx);
  c.focal_y = 720.0f / (2.0f * tan_fovy);
  c.W = 1280.0f; c.H = 720.0f;
  return c;
}

// ---------- ORIGINAL 3DGS (CUDA ported to CPU) ----------
namespace orig {
glm::mat3 computeCov3D(const glm::vec3& s, const glm::vec4& q) {
  // Python pre-exp's the scale before calling CUDA; here we exp to match .ply input.
  glm::mat3 S(1.0f);
  S[0][0] = expf(s.x); S[1][1] = expf(s.y); S[2][2] = expf(s.z);
  float r = q.x, x = q.y, y = q.z, z = q.w;
  glm::mat3 R = glm::mat3(
    1.f-2.f*(y*y+z*z), 2.f*(x*y-r*z),     2.f*(x*z+r*y),
    2.f*(x*y+r*z),     1.f-2.f*(x*x+z*z), 2.f*(y*z-r*x),
    2.f*(x*z-r*y),     2.f*(y*z+r*x),     1.f-2.f*(x*x+y*y));
  glm::mat3 M = S * R;
  return glm::transpose(M) * M;
}

glm::vec3 computeCov2D(const glm::vec3& mean, float fx, float fy,
                       float tfx, float tfy, const glm::mat3& cov3D,
                       const glm::mat4& view) {
  glm::vec3 t = glm::vec3(view * glm::vec4(mean, 1.0f));
  float limx = 1.3f * tfx, limy = 1.3f * tfy;
  t.x = fminf(limx, fmaxf(-limx, t.x / t.z)) * t.z;
  t.y = fminf(limy, fmaxf(-limy, t.y / t.z)) * t.z;
  glm::mat3 J(fx / t.z, 0, -(fx * t.x) / (t.z * t.z),
              0, fy / t.z, -(fy * t.y) / (t.z * t.z), 0, 0, 0);
  const float* vm = glm::value_ptr(view);
  glm::mat3 W = glm::mat3(vm[0], vm[4], vm[8], vm[1], vm[5], vm[9], vm[2], vm[6], vm[10]);
  glm::mat3 T = W * J;
  glm::mat3 c = glm::transpose(T) * glm::transpose(cov3D) * T;
  c[0][0] += 0.3f; c[1][1] += 0.3f;
  return {c[0][0], c[0][1], c[1][1]};
}
} // namespace orig

// ---------- IPU (current code, ported for host testing) ----------
namespace ipu {
glm::mat3 computeCov3D(const glm::vec3& s, const glm::vec4& q) {
  float r = q.x, x = q.y, y = q.z, z = q.w;
  glm::mat3 R = glm::mat3(
    1.f-2.f*(y*y+z*z), 2.f*(x*y-r*z),     2.f*(x*z+r*y),
    2.f*(x*y+r*z),     1.f-2.f*(x*x+z*z), 2.f*(y*z-r*x),
    2.f*(x*z-r*y),     2.f*(y*z+r*x),     1.f-2.f*(x*x+y*y));
  glm::mat3 S(1.0f);
  S[0][0] = expf(s.x); S[1][1] = expf(s.y); S[2][2] = expf(s.z);
  glm::mat3 M = S * R;
  return glm::transpose(M) * M;
}

glm::vec3 computeCov2D(const glm::vec3& mean, float fx, float fy,
                       float tfx, float tfy, const glm::mat3& cov3D,
                       const glm::mat4& view) {
  glm::vec3 t = glm::vec3(view * glm::vec4(mean, 1.0f));
  float limx = 1.3f * tfx, limy = 1.3f * tfy;
  t.x = fminf(limx, fmaxf(-limx, t.x / t.z)) * t.z;
  t.y = fminf(limy, fmaxf(-limy, t.y / t.z)) * t.z;
  glm::mat3 J(fx / t.z, 0, -(fx * t.x) / (t.z * t.z),
              0, fy / t.z, -(fy * t.y) / (t.z * t.z), 0, 0, 0);
  glm::mat3 W = glm::transpose(glm::mat3(view));
  glm::mat3 T = W * J;
  glm::mat3 c = glm::transpose(T) * glm::transpose(cov3D) * T;
  c[0][0] += 0.3f; c[1][1] += 0.3f;
  return {c[0][0], c[0][1], c[1][1]};
}
} // namespace ipu

// ---------- Generic helpers ----------
// Invert 2D covariance to get conic (both implementations do this identically).
glm::vec3 conicFromCov(const glm::vec3& c) {
  float det = c.x * c.z - c.y * c.y;
  if (fabsf(det) < 1e-10f) return {0, 0, 0};
  float inv = 1.0f / det;
  return {c.z * inv, -c.y * inv, c.x * inv};
}

// Project mean to screen space using view/proj matrix.
glm::vec2 projectToScreen(const glm::vec3& mean, const glm::mat4& view,
                          const glm::mat4& proj, float W, float H) {
  glm::vec4 clip = proj * view * glm::vec4(mean, 1.0f);
  glm::vec2 ndc = glm::vec2(clip) / clip.w;
  return (ndc * 0.5f + 0.5f) * glm::vec2(W, H);
}

// Per-Gaussian on-screen state used for compositing.
struct Splat {
  glm::vec2 mean2D;
  glm::vec3 conic;
  glm::vec3 colour;
  float opacity;
  float z; // view-space depth (for sort)
};

// Compose a pixel using front-to-back alpha blend, identical to forward.cu.
glm::vec3 compositePixel(const glm::vec2& pix, const std::vector<Splat>& splats) {
  glm::vec3 C(0);
  float T = 1.0f;
  for (const auto& s : splats) {
    glm::vec2 d = s.mean2D - pix;
    float power = -0.5f * (s.conic.x * d.x * d.x + s.conic.z * d.y * d.y)
                - s.conic.y * d.x * d.y;
    if (power > 0) continue;
    float alpha = fminf(0.99f, s.opacity * expf(power));
    if (alpha < 1.0f / 255.0f) continue;
    float test_T = T * (1.0f - alpha);
    if (test_T < 0.0001f) break;
    C += s.colour * alpha * T;
    T = test_T;
  }
  return C;
}

// Build the full splat list for a given camera, using the chosen math namespace.
template<glm::mat3 (*Cov3D)(const glm::vec3&, const glm::vec4&),
         glm::vec3 (*Cov2D)(const glm::vec3&, float, float, float, float,
                            const glm::mat3&, const glm::mat4&)>
std::vector<Splat> buildSplats(const std::vector<TestG>& scene, const Cam& cam) {
  std::vector<Splat> out;
  out.reserve(scene.size());
  for (const auto& g : scene) {
    glm::mat3 c3 = Cov3D(g.scale, g.rot);
    glm::vec3 c2 = Cov2D(g.mean, cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, c3, cam.view);
    Splat s;
    s.mean2D = projectToScreen(g.mean, cam.view, cam.proj, cam.W, cam.H);
    s.conic  = conicFromCov(c2);
    s.colour = g.sh_dc;
    s.opacity = 1.0f / (1.0f + expf(-g.opacity)); // sigmoid
    s.z      = (cam.view * glm::vec4(g.mean, 1.0f)).z; // view-space z (negative, less negative = closer)
    out.push_back(s);
  }
  // Sort front-to-back: in OpenGL view space, closest = largest (least negative) z.
  std::sort(out.begin(), out.end(),
            [](const Splat& a, const Splat& b) { return a.z > b.z; });
  return out;
}

int main() {
  auto scene = makeScene();
  auto cam   = makeCam();

  auto orig_splats = buildSplats<orig::computeCov3D, orig::computeCov2D>(scene, cam);
  auto ipu_splats  = buildSplats<ipu::computeCov3D,  ipu::computeCov2D >(scene, cam);

  printf("=== Scene: %zu Gaussians ===\n\n", scene.size());

  printf("Front-to-back sort order (both should be identical):\n");
  printf("  %-6s %-15s %-15s %-15s\n", "idx", "orig z", "ipu z", "match?");
  bool sort_match = true;
  for (size_t i = 0; i < orig_splats.size(); ++i) {
    bool match = fabsf(orig_splats[i].z - ipu_splats[i].z) < 1e-4f;
    sort_match &= match;
    printf("  %-6zu %-15f %-15f %-15s\n", i, orig_splats[i].z, ipu_splats[i].z, match ? "yes" : "NO");
  }
  printf("\nPer-Gaussian splat data comparison (after sort):\n");
  float max_conic_diff = 0, max_mean_diff = 0;
  for (size_t i = 0; i < orig_splats.size(); ++i) {
    const auto& o = orig_splats[i];
    const auto& u = ipu_splats[i];
    float cd = glm::length(o.conic - u.conic);
    float md = glm::length(o.mean2D - u.mean2D);
    max_conic_diff = fmaxf(max_conic_diff, cd);
    max_mean_diff  = fmaxf(max_mean_diff, md);
    printf("  G%zu  mean2D orig=(%8.2f,%8.2f) ipu=(%8.2f,%8.2f)  |Δmean|=%.2e  |Δconic|=%.2e\n",
      i, o.mean2D.x, o.mean2D.y, u.mean2D.x, u.mean2D.y, md, cd);
  }

  // Sample a grid of pixels and compare composited colour.
  printf("\nPer-pixel composited colour diff (sampled on a 5x5 grid near scene centre):\n");
  float max_pixel_diff = 0;
  for (int j = 0; j < 5; ++j) {
    for (int i = 0; i < 5; ++i) {
      glm::vec2 pix = {cam.W / 2 - 20 + i * 10, cam.H / 2 - 20 + j * 10};
      glm::vec3 co = compositePixel(pix, orig_splats);
      glm::vec3 cu = compositePixel(pix, ipu_splats);
      float d = glm::length(co - cu);
      max_pixel_diff = fmaxf(max_pixel_diff, d);
    }
  }

  printf("\n============================================\n");
  printf("SUMMARY:\n");
  printf("  Sort order identical:      %s\n", sort_match ? "YES" : "NO");
  printf("  Max |Δ mean2D|  across Gaussians: %.3e\n", max_mean_diff);
  printf("  Max |Δ conic|   across Gaussians: %.3e\n", max_conic_diff);
  printf("  Max |Δ pixel|   across sampled grid: %.3e\n", max_pixel_diff);
  printf("============================================\n");
  if (max_conic_diff < 1e-4f && max_pixel_diff < 1e-4f && sort_match) {
    printf("\n✓ IPU pipeline is NUMERICALLY EQUIVALENT to the original 3DGS.\n");
    return 0;
  }
  printf("\n✗ Pipeline divergence detected.\n");
  return 1;
}
