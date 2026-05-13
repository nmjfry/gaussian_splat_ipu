// Tile-churn measurement experiment for EGSR rebuttal.
// Host-only: no IPU/Poplar needed. Measures how often Gaussians change anchor
// tile between frames under various camera trajectories.
//
// Compile:
//   g++ -std=c++17 -O2 -I include -I external/glm -I external \
//       tools/churn_experiment.cpp -o build/churn_experiment
//
// Usage:
//   ./build/churn_experiment --ply data/point_cloud_10.ply --output churn_results.csv

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <cassert>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <happly.h>

// ---------- Constants matching the IPU pipeline ----------
static constexpr float IMWIDTH  = 1280.0f;
static constexpr float IMHEIGHT = 720.0f;
static constexpr float TILEWIDTH  = 32.0f;
static constexpr float TILEHEIGHT = 20.0f;
static constexpr int TILES_ACROSS = 40;
static constexpr int TILES_DOWN   = 36;
static constexpr int NUM_TILES = TILES_ACROSS * TILES_DOWN; // 1440

// ---------- Bounding box ----------
struct BBox {
  glm::vec3 mn{1e30f}, mx{-1e30f};
  void expand(glm::vec3 p) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
  glm::vec3 centroid() const { return 0.5f * (mn + mx); }
  glm::vec3 diagonal() const { return mx - mn; }
};

// ---------- PLY loading ----------
struct Scene {
  std::vector<glm::vec3> means;
  BBox bb;
};

Scene loadPly(const std::string& path) {
  happly::PLYData ply(path);
  auto x = ply.getElement("vertex").getProperty<float>("x");
  auto y = ply.getElement("vertex").getProperty<float>("y");
  auto z = ply.getElement("vertex").getProperty<float>("z");
  Scene s;
  s.means.resize(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    s.means[i] = {x[i], y[i], z[i]};
    s.bb.expand(s.means[i]);
  }
  // Centre the scene (same as splat.cpp:111-117)
  glm::vec3 c = s.bb.centroid();
  for (auto& m : s.means) m -= c;
  s.bb.mn -= c;
  s.bb.mx -= c;
  return s;
}

// ---------- Camera helpers (matching splat.cpp exactly) ----------

// lookAtBoundingBox from camera.cpp
glm::mat4 lookAtBB(const BBox& bb, glm::vec3 up, float scale) {
  glm::vec3 lookAt = bb.centroid();
  float radius = glm::length(bb.diagonal()) * 0.5f;
  glm::vec3 camPos = lookAt + glm::vec3(0.f, 0.f, scale * radius);
  return glm::lookAt(camPos, lookAt, up);
}

// fitFrustumToBoundingBox from geometry.cpp (3DGS-style, z_sign=+1)
glm::mat4 fitFrustum(const BBox& bb, float fovHalfRad, float aspect) {
  float radius = glm::length(bb.diagonal()) * 0.5f;
  float nearPlane = 0.01f * radius;
  float farPlane = nearPlane + 21.f * radius;
  float tanHalfFovY = std::tan(fovHalfRad);
  float tanHalfFovX = tanHalfFovY * aspect;
  float top    =  tanHalfFovY * nearPlane;
  float bottom = -top;
  float right  =  tanHalfFovX * nearPlane;
  float left   = -right;
  glm::mat4 P(0.f);
  P[0][0] = 2.f * nearPlane / (right - left);
  P[1][1] = 2.f * nearPlane / (top - bottom);
  P[2][0] = (right + left) / (right - left);
  P[2][1] = (top + bottom) / (top - bottom);
  P[2][2] = farPlane / (farPlane - nearPlane);
  P[2][3] = 1.f;
  P[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
  return P;
}

// OpenGL→COLMAP flip: diag(1, -1, -1, 1)
static const glm::mat4 kFlipYZ = glm::mat4(
    glm::vec4( 1.f,  0.f,  0.f, 0.f),
    glm::vec4( 0.f, -1.f,  0.f, 0.f),
    glm::vec4( 0.f,  0.f, -1.f, 0.f),
    glm::vec4( 0.f,  0.f,  0.f, 1.f));

// ---------- Projection + tile assignment ----------

// DGR's ndc2Pix: pixel = ((ndc + 1) * S - 1) * 0.5  (viewport.hpp:18-19)
glm::vec2 ndc2Pix(glm::vec2 ndc) {
  return glm::vec2(
    ((ndc.x + 1.f) * IMWIDTH  - 1.f) * 0.5f,
    ((ndc.y + 1.f) * IMHEIGHT - 1.f) * 0.5f
  );
}

// pixCoordToTile from tile_config.hpp:43-53
int pixCoordToTile(float row, float col) {
  float r = std::nearbyint(row);
  float c = std::nearbyint(col);
  int tileCol = (int)std::floor(c / TILEWIDTH);
  int tileRow = (int)std::floor(r / TILEHEIGHT);
  if (tileCol < 0 || tileCol >= TILES_ACROSS || tileRow < 0 || tileRow >= TILES_DOWN)
    return -1;
  return tileRow * TILES_ACROSS + tileCol;
}

// Project a single Gaussian mean and return its anchor tile (-1 if culled)
int projectAndAssign(glm::vec3 mean, const glm::mat4& viewProj, const glm::mat4& view) {
  // View-space z for near-plane cull
  glm::vec4 viewPos = view * glm::vec4(mean, 1.f);
  if (viewPos.z <= 0.2f) return -1; // near-plane cull (COLMAP: visible z > 0)

  glm::vec4 clip = viewProj * glm::vec4(mean, 1.f);
  if (std::abs(clip.w) < 1e-8f) return -1;
  glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
  glm::vec2 pix = ndc2Pix(ndc);

  // Off-screen cull
  if (pix.x < 0 || pix.x >= IMWIDTH || pix.y < 0 || pix.y >= IMHEIGHT)
    return -1;

  return pixCoordToTile(pix.y, pix.x); // row=y, col=x
}

// ---------- Tile grid helpers ----------

struct TileXY { int x, y; };

TileXY tileToXY(int tid) {
  return {tid % TILES_ACROSS, tid / TILES_ACROSS};
}

int manhattanDist(int t1, int t2) {
  auto a = tileToXY(t1), b = tileToXY(t2);
  return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

// ---------- Trajectory generators ----------

struct Trajectory {
  std::string name;
  int numFrames;
  // Returns the OpenGL-convention view matrix for frame i (pre-COLMAP-flip)
  std::function<glm::mat4(int)> pose;
};

std::vector<Trajectory> makeTrajectories(const BBox& bb) {
  glm::vec3 centre = bb.centroid();
  float radius = glm::length(bb.diagonal()) * 0.5f;
  glm::vec3 up(0.f, 1.f, 0.f);

  // Base view: camera at +Z, looking at centre (same as splat.cpp)
  glm::mat4 baseView = lookAtBB(bb, up, 2.f);

  std::vector<Trajectory> out;

  // 1. Static
  out.push_back({"static", 100, [=](int) { return baseView; }});

  // 2. Orbits at varying angular velocities
  for (float omega : {0.1f, 0.5f, 2.0f, 5.0f, 20.0f}) {
    char name[64];
    snprintf(name, sizeof(name), "orbit_%.1fdeg", omega);
    out.push_back({name, 300, [=](int frame) {
      float angle = glm::radians(omega * frame);
      glm::vec3 camPos = centre + glm::vec3(
        std::sin(angle) * 2.f * radius,
        0.f,
        std::cos(angle) * 2.f * radius
      );
      return glm::lookAt(camPos, centre, up);
    }});
  }

  // 3. Pure rotation about camera center (yaw)
  out.push_back({"pure_rotation_1deg", 300, [=](int frame) {
    glm::vec3 camPos = centre + glm::vec3(0.f, 0.f, 2.f * radius);
    float angle = glm::radians(1.f * frame);
    glm::vec3 target = camPos + glm::vec3(std::sin(angle), 0.f, -std::cos(angle));
    return glm::lookAt(camPos, target, up);
  }});

  // 4. Pure translation (forward along Z)
  out.push_back({"pure_translation", 300, [=](int frame) {
    // Move slowly forward: ~1 pixel/frame equivalent at median depth
    float step = radius * 0.001f * frame;
    glm::vec3 camPos = centre + glm::vec3(0.f, 0.f, 2.f * radius) - glm::vec3(0.f, 0.f, step);
    return glm::lookAt(camPos, centre, up);
  }});

  // 5. Random teleport
  out.push_back({"random_teleport", 100, [=](int frame) {
    std::mt19937 rng(42 + frame);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    glm::vec3 dir(dist(rng), dist(rng), dist(rng));
    dir = glm::normalize(dir);
    glm::vec3 camPos = centre + dir * 2.f * radius;
    return glm::lookAt(camPos, centre, up);
  }});

  return out;
}

// ---------- Churn metrics ----------

struct FrameMetrics {
  std::string trajectory;
  int frame;
  int n_visible;
  int n_stable, n_moved, n_entered, n_exited;
  float churn_rate;
  float work_ratio;
  float mean_hops;
  float p95_hops;
};

FrameMetrics computeMetrics(
    const std::string& trajName, int frame,
    const std::vector<int>& current,
    const std::vector<int>& previous)
{
  int N = (int)current.size();
  assert(N == (int)previous.size());

  int n_stable = 0, n_moved = 0, n_entered = 0, n_exited = 0;
  std::vector<int> hopDistances;

  for (int i = 0; i < N; i++) {
    bool visPrev = previous[i] >= 0;
    bool visNow  = current[i]  >= 0;

    if (visPrev && visNow) {
      if (current[i] == previous[i]) {
        n_stable++;
      } else {
        n_moved++;
        hopDistances.push_back(manhattanDist(previous[i], current[i]));
      }
    } else if (!visPrev && visNow) {
      n_entered++;
    } else if (visPrev && !visNow) {
      n_exited++;
    }
  }

  int n_visible = n_stable + n_moved + n_entered;
  int intersection = n_stable + n_moved;
  float churn = intersection > 0 ? (float)n_moved / intersection : 0.f;
  int denom = n_stable + n_moved + n_entered;
  float work = denom > 0 ? (float)(n_moved + n_entered) / denom : 0.f;

  float meanH = 0.f, p95H = 0.f;
  if (!hopDistances.empty()) {
    meanH = std::accumulate(hopDistances.begin(), hopDistances.end(), 0.f) / hopDistances.size();
    std::sort(hopDistances.begin(), hopDistances.end());
    int idx95 = (int)(0.95f * (hopDistances.size() - 1));
    p95H = (float)hopDistances[idx95];
  }

  // Sanity checks
  int nVisPrev = 0;
  for (int t : previous) if (t >= 0) nVisPrev++;
  assert(n_stable + n_moved + n_exited == nVisPrev);

  return {trajName, frame, n_visible, n_stable, n_moved, n_entered, n_exited,
          churn, work, meanH, p95H};
}

// ---------- CSV output ----------

void writeHeader(FILE* f) {
  fprintf(f, "trajectory,frame,n_visible,n_stable,n_moved,n_entered,n_exited,"
             "churn_rate,work_ratio,mean_hops,p95_hops\n");
}

void writeRow(FILE* f, const FrameMetrics& m) {
  fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.1f\n",
          m.trajectory.c_str(), m.frame, m.n_visible,
          m.n_stable, m.n_moved, m.n_entered, m.n_exited,
          m.churn_rate, m.work_ratio, m.mean_hops, m.p95_hops);
}

// ---------- Main ----------

int main(int argc, char** argv) {
  std::string plyPath, outputPath = "churn_results.csv";
  float fovHalfRad = glm::radians(30.f); // 60° full FOV (half = 30°)

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--ply" && i + 1 < argc)
      plyPath = argv[++i];
    else if (std::string(argv[i]) == "--output" && i + 1 < argc)
      outputPath = argv[++i];
    else if (std::string(argv[i]) == "--fov-half-rad" && i + 1 < argc)
      fovHalfRad = std::stof(argv[++i]);
  }

  if (plyPath.empty()) {
    fprintf(stderr, "Usage: %s --ply <path.ply> [--output <out.csv>] [--fov-half-rad <rad>]\n", argv[0]);
    return 1;
  }

  printf("Loading PLY: %s\n", plyPath.c_str());
  Scene scene = loadPly(plyPath);
  int N = (int)scene.means.size();
  printf("Loaded %d Gaussians\n", N);
  printf("Scene BB: (%.3f,%.3f,%.3f) to (%.3f,%.3f,%.3f)\n",
         scene.bb.mn.x, scene.bb.mn.y, scene.bb.mn.z,
         scene.bb.mx.x, scene.bb.mx.y, scene.bb.mx.z);

  float aspect = IMWIDTH / IMHEIGHT;

  // Build projection from camera-space BB (same as splat.cpp:290-296)
  glm::mat4 baseView = lookAtBB(scene.bb, glm::vec3(0,1,0), 2.f);
  BBox bbCam;
  bbCam.expand(glm::vec3(baseView * glm::vec4(scene.bb.mn, 1.f)));
  bbCam.expand(glm::vec3(baseView * glm::vec4(scene.bb.mx, 1.f)));
  glm::mat4 proj = fitFrustum(bbCam, fovHalfRad, aspect);

  auto trajectories = makeTrajectories(scene.bb);

  FILE* out = fopen(outputPath.c_str(), "w");
  if (!out) { fprintf(stderr, "Cannot open %s\n", outputPath.c_str()); return 1; }
  writeHeader(out);

  std::vector<int> current(N), previous(N, -1);

  for (auto& traj : trajectories) {
    printf("Running trajectory: %s (%d frames)\n", traj.name.c_str(), traj.numFrames);
    std::fill(previous.begin(), previous.end(), -1);

    for (int frame = 0; frame < traj.numFrames; frame++) {
      glm::mat4 viewGL = traj.pose(frame);
      glm::mat4 view = kFlipYZ * viewGL; // OpenGL→COLMAP
      glm::mat4 vp = proj * view;

      // Project all Gaussians
      for (int i = 0; i < N; i++) {
        current[i] = projectAndAssign(scene.means[i], vp, view);
      }

      if (frame > 0) {
        auto m = computeMetrics(traj.name, frame, current, previous);
        writeRow(out, m);

        // Print summary for first and last frames
        if (frame == 1 || frame == traj.numFrames - 1) {
          printf("  frame %3d: visible=%d stable=%d moved=%d entered=%d exited=%d "
                 "churn=%.4f work=%.4f mean_hops=%.1f\n",
                 frame, m.n_visible, m.n_stable, m.n_moved, m.n_entered, m.n_exited,
                 m.churn_rate, m.work_ratio, m.mean_hops);
        }
      } else {
        // Frame 0: write an initial row (all entered)
        int vis = 0;
        for (int t : current) if (t >= 0) vis++;
        fprintf(out, "%s,0,%d,0,0,%d,0,0.000000,1.000000,0.000,0.0\n",
                traj.name.c_str(), vis, vis);
      }

      std::copy(current.begin(), current.end(), previous.begin());
    }

    // Print trajectory summary
    {
      // Rewind and compute mean churn across all frames
      // (simpler: just accumulate during the loop — but we already wrote CSV so
      //  re-read is wasteful. Instead track running stats.)
    }
  }

  fclose(out);
  printf("\nResults written to: %s\n", outputPath.c_str());

  // Print a quick summary table
  printf("\n--- Summary (mean churn per trajectory) ---\n");
  printf("%-25s %8s %8s %8s\n", "Trajectory", "Churn", "WorkRatio", "MeanHops");

  // Re-read CSV to summarize
  std::ifstream fin(outputPath);
  std::string line;
  std::getline(fin, line); // skip header
  std::map<std::string, std::vector<float>> churnByTraj, workByTraj, hopsByTraj;
  while (std::getline(fin, line)) {
    std::stringstream ss(line);
    std::string tname;
    int frame_, vis_, sta_, mov_, ent_, exi_;
    float churn_, work_, mhops_, p95_;
    char comma;
    std::getline(ss, tname, ',');
    ss >> frame_ >> comma >> vis_ >> comma >> sta_ >> comma >> mov_ >> comma
       >> ent_ >> comma >> exi_ >> comma >> churn_ >> comma >> work_ >> comma
       >> mhops_ >> comma >> p95_;
    if (frame_ > 0) { // skip frame 0
      churnByTraj[tname].push_back(churn_);
      workByTraj[tname].push_back(work_);
      hopsByTraj[tname].push_back(mhops_);
    }
  }
  for (auto& [name, churns] : churnByTraj) {
    float mc = std::accumulate(churns.begin(), churns.end(), 0.f) / churns.size();
    auto& works = workByTraj[name];
    float mw = std::accumulate(works.begin(), works.end(), 0.f) / works.size();
    auto& hops = hopsByTraj[name];
    float mh = std::accumulate(hops.begin(), hops.end(), 0.f) / hops.size();
    printf("%-25s %8.4f %8.4f %8.2f\n", name.c_str(), mc, mw, mh);
  }

  return 0;
}
