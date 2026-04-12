// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

  // In COLMAP view space (used after the OpenGL->COLMAP conversion in splat.cpp),
  // the camera looks down +Z, so visible points have positive view_z. The
  // nearest scene face has the smallest positive view_z. For the bounding box
  // flipped into COLMAP view, bb.min.z becomes the nearest; use a safe fraction
  // of the scene radius as the near plane.
  float nearPlane = 0.01f * radius;
  float farPlane = nearPlane + 21.f * radius;

  // 3DGS-style projection matrix (z_sign = +1). Matches
  // graphdeco-inria/gaussian-splatting's getProjectionMatrix() exactly.
  // fovRadians is HALF-FOV (already halved by InterfaceServer).
  const float tanHalfFovY = std::tan(fovRadians);
  const float tanHalfFovX = tanHalfFovY * aspectRatio;
  const float top    =  tanHalfFovY * nearPlane;
  const float bottom = -top;
  const float right  =  tanHalfFovX * nearPlane;
  const float left   = -right;

  // GLM stores matrices column-major: P[col][row].
  // Python reference (row-major) indices map to GLM as P[col][row] = python[row][col].
  glm::mat4 P(0.f);
  P[0][0] = 2.f * nearPlane / (right - left);
  P[1][1] = 2.f * nearPlane / (top - bottom);
  P[2][0] = (right + left) / (right - left);
  P[2][1] = (top + bottom) / (top - bottom);
  P[2][2] = farPlane / (farPlane - nearPlane);
  P[2][3] = 1.f;                                         // z_sign
  P[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
  return P;
}

} // end of namespace splat
