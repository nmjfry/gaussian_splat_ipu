// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

  // In view space (after glm::lookAt), scene is at negative Z.
  // bb.max.z is closest to camera (least negative).
  float nearFaceDistance = -bb.max.z;
  float nearPlane = glm::max(0.01f * radius, nearFaceDistance * 0.5f);
  float farPlane = nearPlane + 21.f * radius;

  // Standard OpenGL projection (z_sign = -1, matching glm::lookAt convention).
  // fovRadians is HALF-FOV (already halved by InterfaceServer), so multiply by 2.
  float fullFovY = fovRadians * 2.0f;
  return glm::perspective(fullFovY, aspectRatio, nearPlane, farPlane);
}

} // end of namespace splat
