// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/camera.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace splat {

// Return the model view matrix for a camera that is scale times the bounding radius
// away from the box, down the +ve z-axis, and looking at the centroid of the box:
glm::mat4 lookAtBoundingBox(const Bounds3f& bb, const glm::vec3& up, float scale) {
  auto lookAtPoint = bb.centroid();
  auto radius = glm::length(bb.diagonal()) * .5f;
  auto cameraPosition = lookAtPoint - glm::vec3(0.f, 0.f, scale * radius);
  return glm::lookAt(cameraPosition, lookAtPoint, up);
}

} // end of namespace splat
