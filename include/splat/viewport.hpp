

#include <glm/glm.hpp>

namespace splat {

struct Viewport {
  Viewport(float x, float y, float width, float height)
    : spec(x, y, width, height) {}

  // Match diff-gaussian-rasterization's ndc2Pix exactly:
  //   pixel = ((ndc + 1) * S - 1) * 0.5
  // This treats integer pixel coord i as the CENTRE of that pixel (so pixel 0
  // sits at ndc = (1 - S)/S, not at ndc = -1). Without this, the Gaussian
  // means land half a pixel off from where DGR puts them.
  glm::vec2 ndcToViewport(glm::vec4 ndc) const {
    glm::vec2 vp;
    vp.x = ((ndc.x + 1.f) * spec[2] - 1.f) * 0.5f + spec[0];
    vp.y = ((ndc.y + 1.f) * spec[3] - 1.f) * 0.5f + spec[1];
    return vp;
  }

  glm::vec2 clipSpaceToViewport(glm::vec4 cs) const {
    // Perspective divide, then ndc2Pix (DGR convention).
    glm::vec2 ndc(cs.x / cs.w, cs.y / cs.w);
    glm::vec2 vp;
    vp.x = ((ndc.x + 1.f) * spec[2] - 1.f) * 0.5f + spec[0];
    vp.y = ((ndc.y + 1.f) * spec[3] - 1.f) * 0.5f + spec[1];
    return vp;
  }

  // Legacy helper — kept for any callers still using it (e.g. CPU point
  // rasteriser); matches the old half-open [0, S] mapping.
  glm::vec2 viewportTransform(glm::vec2 v) const {
    v.x *= spec[2];
    v.y *= spec[3];
    v.x += spec[0];
    v.y += spec[1];
    return v;
  }

  glm::vec4 spec;
};

} // end of namespace splat