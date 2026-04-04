// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

  float nearFaceDistance = -bb.min.z;  // positive distance to nearest scene face
  float nearPlane = glm::max(0.01f * radius, nearFaceDistance * 0.5f);
  float farPlane = nearPlane + 21.f * radius;

  // Build projection matrix matching the original 3DGS convention:
  // Uses z_sign = +1.0 (NOT standard OpenGL which uses -1).
  float fovY = fovRadians;
  float fovX = 2.f * atanf(tanf(fovY / 2.f) * aspectRatio);
  float tanHalfFovY = tanf(fovY / 2.f);
  float tanHalfFovX = tanf(fovX / 2.f);

  float top    =  tanHalfFovY * nearPlane;
  float bottom = -top;
  float right  =  tanHalfFovX * nearPlane;
  float left   = -right;

  // The original 3DGS Python builds P in row-major [row][col] then
  // transposes to column-major for the GPU. GLM uses column-major P[col][row].
  // So Python P[row][col] becomes GLM P[col][row] after transpose.
  //
  // Python (row-major):           GLM (column-major) P[col][row]:
  //   P[0,0] = 2n/(r-l)     -->   P[0][0] = 2n/(r-l)
  //   P[1,1] = 2n/(t-b)     -->   P[1][1] = 2n/(t-b)
  //   P[0,2] = (r+l)/(r-l)  -->   P[2][0] = (r+l)/(r-l)
  //   P[1,2] = (t+b)/(t-b)  -->   P[2][1] = (t+b)/(t-b)
  //   P[2,2] = f/(f-n)      -->   P[2][2] = f/(f-n)
  //   P[2,3] = -fn/(f-n)    -->   P[3][2] = -fn/(f-n)
  //   P[3,2] = 1.0           -->   P[2][3] = 1.0
  glm::mat4 P(0.f);
  P[0][0] = 2.0f * nearPlane / (right - left);
  P[1][1] = 2.0f * nearPlane / (top - bottom);
  P[2][0] = (right + left) / (right - left);
  P[2][1] = (top + bottom) / (top - bottom);
  P[2][2] = farPlane / (farPlane - nearPlane);
  P[2][3] = 1.0f;
  P[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
  return P;
}

} // end of namespace splat
