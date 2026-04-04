// Comparison test: IPU 3DGS math vs Original 3DGS (CUDA forward.cu) math
// Compile: g++ -std=c++17 -I../include -I../external/glm tests/test_comparison.cpp -o test_comparison
//
// This runs both implementations on the same Gaussian and camera,
// printing intermediate values at each stage to identify divergence.

#include <cstdio>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

// ============================================================
// Test Gaussian (pick one from your .ply or hardcode)
// ============================================================
struct TestGaussian {
  glm::vec3 mean;
  glm::vec3 scale;     // log-space (as stored in .ply)
  glm::vec4 rot;       // rot_0, rot_1, rot_2, rot_3 from .ply
  float opacity;
  glm::vec3 sh_dc;     // f_dc_0, f_dc_1, f_dc_2
};

// A simple test Gaussian — replace with real values from your .ply
TestGaussian testGaussian() {
  return {
    .mean    = {0.5f, -0.3f, -2.0f},
    .scale   = {-3.0f, -4.0f, -3.5f},   // log-space
    .rot     = {0.9f, 0.1f, 0.2f, 0.3f}, // rot_0=w, rot_1=x, rot_2=y, rot_3=z (3DGS convention)
    .opacity = 0.8f,
    .sh_dc   = {0.1f, 0.2f, 0.15f},
  };
}

// A simple test camera
struct TestCamera {
  glm::mat4 viewMatrix;
  glm::mat4 projMatrix;
  float fovX_rad;
  float fovY_rad;
  float imageW, imageH;
};

TestCamera testCamera() {
  float fovY = glm::radians(60.0f);
  float aspect = 1280.0f / 720.0f;
  float fovX = 2.0f * atanf(tanf(fovY / 2.0f) * aspect);

  glm::mat4 view = glm::lookAt(
    glm::vec3(0.0f, 0.0f, 3.0f),   // eye
    glm::vec3(0.0f, 0.0f, 0.0f),   // center
    glm::vec3(0.0f, 1.0f, 0.0f)    // up
  );

  glm::mat4 proj = glm::perspective(fovY, aspect, 0.01f, 100.0f);

  return { view, proj, fovX, fovY, 1280.0f, 720.0f };
}

void printMat3(const char* name, const glm::mat3& m) {
  printf("  %s:\n", name);
  for (int r = 0; r < 3; r++)
    printf("    [%12.6f %12.6f %12.6f]\n", m[0][r], m[1][r], m[2][r]);
}

void printVec3(const char* name, const glm::vec3& v) {
  printf("  %s: [%12.6f %12.6f %12.6f]\n", name, v.x, v.y, v.z);
}

void printCov2D(const char* name, float a, float b, float c) {
  printf("  %s: [%12.6f %12.6f %12.6f]  (cov00, cov01, cov11)\n", name, a, b, c);
}

// ============================================================
// ORIGINAL 3DGS (from forward.cu, ported to CPU)
// ============================================================
namespace original {

glm::mat3 computeCov3D(const glm::vec3& scale, const glm::vec4& rot) {
  // Original uses raw (already exp'd) scale. We exp() here since .ply stores log-space.
  glm::mat3 S(1.0f);
  S[0][0] = expf(scale.x);
  S[1][1] = expf(scale.y);
  S[2][2] = expf(scale.z);

  // Quaternion: rot = (w, x, y, z) in 3DGS .ply convention
  float r = rot.x;  // w (real part)
  float x = rot.y;  // x
  float y = rot.z;  // y
  float z = rot.w;  // z

  // Rotation matrix — this is what the CUDA code builds manually.
  // NOTE: GLM is column-major, so glm::mat3(col0, col1, col2)
  glm::mat3 R = glm::mat3(
    1.f - 2.f*(y*y + z*z), 2.f*(x*y - r*z),       2.f*(x*z + r*y),        // column 0
    2.f*(x*y + r*z),       1.f - 2.f*(x*x + z*z),  2.f*(y*z - r*x),        // column 1
    2.f*(x*z - r*y),       2.f*(y*z + r*x),         1.f - 2.f*(x*x + y*y)   // column 2
  );

  glm::mat3 M = S * R;
  glm::mat3 Sigma = glm::transpose(M) * M;
  return Sigma;
}

// Original computeCov2D from forward.cu
// viewmatrix is passed as a float* in row-major order in the CUDA code.
// Here we pass the GLM column-major mat4 and extract W the same way the CUDA code does.
glm::vec3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y,
                        float tan_fovx, float tan_fovy,
                        const glm::mat3& cov3D, const glm::mat4& viewmatrix) {
  // Transform mean to view space
  glm::vec3 t = glm::vec3(viewmatrix * glm::vec4(mean, 1.0f));

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = fminf(limx, fmaxf(-limx, txtz)) * t.z;
  t.y = fminf(limy, fmaxf(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(
    focal_x / t.z, 0.0f,          -(focal_x * t.x) / (t.z * t.z),
    0.0f,          focal_y / t.z,  -(focal_y * t.y) / (t.z * t.z),
    0, 0, 0);

  // CUDA code extracts W from the viewmatrix stored as float[16] in ROW-MAJOR order:
  //   W = mat3(vm[0], vm[4], vm[8],   <-- column 0
  //            vm[1], vm[5], vm[9],   <-- column 1
  //            vm[2], vm[6], vm[10])  <-- column 2
  //
  // Since the CUDA code stores viewmatrix row-major but GLM is column-major,
  // vm[0]=col0.x, vm[1]=col0.y, vm[2]=col0.z, vm[3]=col0.w (in GLM memory)
  // vm[4]=col1.x, vm[5]=col1.y, vm[6]=col1.z, vm[7]=col1.w
  // etc.
  //
  // So the CUDA W columns are:
  //   col0 = (vm[0], vm[4], vm[8])  = (viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0])
  //   col1 = (vm[1], vm[5], vm[9])  = (viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1])
  //   col2 = (vm[2], vm[6], vm[10]) = (viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2])
  //
  // This is the TRANSPOSE of glm::mat3(viewmatrix):
  const float* vm = glm::value_ptr(viewmatrix);
  glm::mat3 W = glm::mat3(
    vm[0], vm[4], vm[8],
    vm[1], vm[5], vm[9],
    vm[2], vm[6], vm[10]
  );

  glm::mat3 T = W * J;

  // The CUDA code does: cov = transpose(T) * transpose(Vrk) * T
  // Since Vrk (cov3D) is symmetric, transpose(Vrk) == Vrk, so this is:
  // cov = transpose(T) * cov3D * T
  glm::mat3 cov = glm::transpose(T) * glm::transpose(cov3D) * T;

  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return { cov[0][0], cov[0][1], cov[1][1] };
}

} // namespace original

// ============================================================
// IPU implementation (from ipu_geometry.hpp)
// ============================================================
namespace ipu {

glm::mat3 computeCov3D(const glm::vec3& scale, const glm::vec4& rot) {
  // FIXED: matches original 3DGS: M = S * R, Sigma = M^T * M
  float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
  glm::mat3 R = glm::mat3(
    1.f - 2.f*(y*y + z*z), 2.f*(x*y - r*z),       2.f*(x*z + r*y),
    2.f*(x*y + r*z),       1.f - 2.f*(x*x + z*z),  2.f*(y*z - r*x),
    2.f*(x*z - r*y),       2.f*(y*z + r*x),         1.f - 2.f*(x*x + y*y)
  );
  glm::mat3 S(1.0f);
  S[0][0] = expf(scale.x);
  S[1][1] = expf(scale.y);
  S[2][2] = expf(scale.z);
  glm::mat3 M = S * R;
  return glm::transpose(M) * M;
}

glm::vec3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y,
                        float tan_fovx, float tan_fovy,
                        const glm::mat3& cov3D, const glm::mat4& viewmatrix) {
  glm::vec3 t = glm::vec3(viewmatrix * glm::vec4(mean, 1.0f));

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = fminf(limx, fmaxf(-limx, txtz)) * t.z;
  t.y = fminf(limy, fmaxf(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(
    focal_x / t.z, 0.0f,          -(focal_x * t.x) / (t.z * t.z),
    0.0f,          focal_y / t.z,  -(focal_y * t.y) / (t.z * t.z),
    0, 0, 0);

  // FIXED: W = transpose(glm::mat3(viewmatrix)) to match original CUDA extraction
  glm::mat3 W = glm::transpose(glm::mat3(viewmatrix));
  glm::mat3 T = W * J;

  // FIXED: cov = transpose(T) * transpose(cov3D) * T (matches original)
  glm::mat3 cov = glm::transpose(T) * glm::transpose(cov3D) * T;

  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return { cov[0][0], cov[0][1], cov[1][1] };
}

} // namespace ipu

// ============================================================
// Conic and alpha comparison
// ============================================================
void compareConic(const glm::vec3& cov2D_orig, const glm::vec3& cov2D_ipu) {
  printf("\n=== Stage 5: Conic (inverse of 2D covariance) ===\n");
  auto computeConic = [](const glm::vec3& c) -> glm::vec3 {
    float det = c.x * c.z - c.y * c.y;
    if (fabsf(det) < 1e-10f) return {0,0,0};
    float inv = 1.0f / det;
    return { c.z * inv, -c.y * inv, c.x * inv };
  };

  auto co = computeConic(cov2D_orig);
  auto ci = computeConic(cov2D_ipu);
  printf("  Original conic: [%12.6f %12.6f %12.6f]\n", co.x, co.y, co.z);
  printf("  IPU conic:      [%12.6f %12.6f %12.6f]\n", ci.x, ci.y, ci.z);
  printf("  Diff:           [%12.6f %12.6f %12.6f]\n", co.x-ci.x, co.y-ci.y, co.z-ci.z);
}

void compareAlpha(const glm::vec3& conic, const glm::vec2& mean2D, float opacity,
                  float testPixelX, float testPixelY) {
  printf("\n=== Stage 6: Per-pixel alpha at (%.0f, %.0f) ===\n", testPixelX, testPixelY);
  float dx = testPixelX - mean2D.x;
  float dy = testPixelY - mean2D.y;
  float power = -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
  float alpha = fminf(0.99f, opacity * expf(power));
  printf("  d = (%f, %f)\n", dx, dy);
  printf("  power = %f\n", power);
  printf("  alpha = %f\n", alpha);
}

// ============================================================
// Main
// ============================================================
int main() {
  auto g = testGaussian();
  auto cam = testCamera();

  float tan_fovx = tanf(cam.fovX_rad / 2.0f);
  float tan_fovy = tanf(cam.fovY_rad / 2.0f);
  float focal_x = cam.imageW / (2.0f * tan_fovx);
  float focal_y = cam.imageH / (2.0f * tan_fovy);

  printf("=== Camera Parameters ===\n");
  printf("  focal: (%f, %f)\n", focal_x, focal_y);
  printf("  tan_fov: (%f, %f)\n", tan_fovx, tan_fovy);
  printf("  image: (%.0f x %.0f)\n", cam.imageW, cam.imageH);

  // ---- Stage 1: View-space transform ----
  printf("\n=== Stage 1: View-space transform ===\n");
  glm::vec3 t_orig = glm::vec3(cam.viewMatrix * glm::vec4(g.mean, 1.0f));
  printVec3("View-space mean", t_orig);

  // ---- Stage 2: ComputeCov3D ----
  printf("\n=== Stage 2: ComputeCov3D ===\n");
  auto cov3D_orig = original::computeCov3D(g.scale, g.rot);
  auto cov3D_ipu  = ipu::computeCov3D(g.scale, g.rot);
  printMat3("Original Cov3D", cov3D_orig);
  printMat3("IPU Cov3D", cov3D_ipu);
  auto diff3D = cov3D_orig - cov3D_ipu;
  printMat3("Diff (orig - ipu)", diff3D);

  // ---- Stage 3: ComputeCov2D ----
  printf("\n=== Stage 3: ComputeCov2D ===\n");
  printf("  (includes Jacobian, W extraction, T computation)\n");

  // Also print the W matrix difference
  const float* vm = glm::value_ptr(cam.viewMatrix);
  glm::mat3 W_cuda = glm::mat3(vm[0], vm[4], vm[8], vm[1], vm[5], vm[9], vm[2], vm[6], vm[10]);
  glm::mat3 W_ipu = glm::mat3(cam.viewMatrix);
  printMat3("Original W (CUDA extraction)", W_cuda);
  printMat3("IPU W (glm::mat3(viewmatrix))", W_ipu);
  printMat3("Diff W", W_cuda - W_ipu);

  auto cov2D_orig = original::computeCov2D(g.mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_orig, cam.viewMatrix);
  auto cov2D_ipu  = ipu::computeCov2D(g.mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_ipu, cam.viewMatrix);
  printCov2D("Original cov2D", cov2D_orig.x, cov2D_orig.y, cov2D_orig.z);
  printCov2D("IPU cov2D",      cov2D_ipu.x, cov2D_ipu.y, cov2D_ipu.z);
  printCov2D("Diff",            cov2D_orig.x - cov2D_ipu.x, cov2D_orig.y - cov2D_ipu.y, cov2D_orig.z - cov2D_ipu.z);

  // ---- Stage 4: Bounding box (same method, skip) ----
  printf("\n=== Stage 4: Bounding box ===\n");
  printf("  (Both use 3*sqrt(max_eigenvalue) — skipping, identical by construction)\n");

  // ---- Stage 5: Conic ----
  compareConic(cov2D_orig, cov2D_ipu);

  // ---- Stage 6: Per-pixel alpha ----
  // Project mean to screen for a test pixel
  glm::vec4 clip = cam.projMatrix * cam.viewMatrix * glm::vec4(g.mean, 1.0f);
  glm::vec2 ndc = glm::vec2(clip) / clip.w;
  glm::vec2 screen = (ndc * 0.5f + 0.5f) * glm::vec2(cam.imageW, cam.imageH);
  printf("\n  Projected screen mean: (%.1f, %.1f)\n", screen.x, screen.y);

  // Compute alpha at a pixel offset of (5, 3) from the mean
  float sigmoid_opacity = 1.0f / (1.0f + expf(-g.opacity));

  auto conic_orig = [&]() -> glm::vec3 {
    float det = cov2D_orig.x * cov2D_orig.z - cov2D_orig.y * cov2D_orig.y;
    float inv = 1.0f / det;
    return { cov2D_orig.z * inv, -cov2D_orig.y * inv, cov2D_orig.x * inv };
  }();

  auto conic_ipu = [&]() -> glm::vec3 {
    float det = cov2D_ipu.x * cov2D_ipu.z - cov2D_ipu.y * cov2D_ipu.y;
    float inv = 1.0f / det;
    return { cov2D_ipu.z * inv, -cov2D_ipu.y * inv, cov2D_ipu.x * inv };
  }();

  printf("\n  Using sigmoid(opacity) = %f\n", sigmoid_opacity);
  printf("\n  --- Original ---\n");
  compareAlpha(conic_orig, screen, sigmoid_opacity, screen.x + 5, screen.y + 3);
  printf("\n  --- IPU ---\n");
  compareAlpha(conic_ipu, screen, sigmoid_opacity, screen.x + 5, screen.y + 3);

  // ---- Summary of known differences ----
  printf("\n============================================\n");
  printf("KNOWN DIFFERENCES TO INVESTIGATE:\n");
  printf("============================================\n");
  printf("\n1. ComputeCov3D formula:\n");
  printf("   Original: M = S * R, Sigma = M^T * M = R^T * S^2 * R\n");
  printf("   IPU:      Sigma = R * S * S^T * R^T = R * S^2 * R^T\n");
  printf("   These differ unless R is identity. The original uses R^T...R,\n");
  printf("   your code uses R...R^T. Additionally, the original builds R\n");
  printf("   manually from quaternion components, while yours uses\n");
  printf("   glm::mat3(glm::normalize(q)) which may produce R^T.\n");
  printf("   -> Check if the two R matrices are transposes of each other.\n");

  printf("\n2. W extraction in ComputeCov2D:\n");
  printf("   Original: manually extracts from row-major float[16] -> effectively transpose(glm::mat3(vm))\n");
  printf("   IPU:      W = glm::mat3(viewmatrix)\n");
  printf("   These are transposes of each other! This means T = W*J differs.\n");
  printf("   -> Your W is the transpose of what the original uses.\n");

  printf("\n3. Opacity:\n");
  printf("   Original stores raw opacity, applies sigmoid in the shader.\n");
  printf("   Your code stores raw opacity in .ply but check if sigmoid is\n");
  printf("   applied before or during rendering.\n");

  return 0;
}
