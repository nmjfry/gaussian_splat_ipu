// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

  float nearFaceDistance = -bb.min.z;  // positive distance to nearest scene face
  float nearPlane = glm::max(0.01f * radius, nearFaceDistance * 0.5f);
  float farPlane = nearPlane + 21.f * radius;

  // Use the actual FOV to build the projection (matches standard 3DGS):
  float fovY = fovRadians;
  return glm::perspective(fovY, aspectRatio, nearPlane, farPlane);
}

} // end of namespace splat
