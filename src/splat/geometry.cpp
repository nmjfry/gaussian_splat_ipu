// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

  // bb is in camera space (view matrix applied). Due to the z-flip in the view
  // matrix, bb.min.z holds the NEAREST z-extent (least negative, closest to camera).
  // The near plane must be placed in front of that face (smaller positive value).
  // radius/tan(fov) is wrong here: for any FOV < 60deg it places the near plane
  // beyond the scene centroid, clipping everything.
  float nearFaceDistance = -bb.min.z;  // positive distance to nearest scene face
  float nearPlane = glm::max(0.01f * radius, nearFaceDistance * 0.5f);
  float farPlane = nearPlane + 21.f * radius;

  float halfWidth = radius * aspectRatio;
  float halfHeight = radius;

  return glm::frustum(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);
}

} // end of namespace splat
