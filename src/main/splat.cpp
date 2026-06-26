// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include "glm/matrix.hpp"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ipu/options.hpp>
#include <ipu/ipu_utils.hpp>
#include <ipu/io_utils.hpp>
#include <splat/camera.hpp>

#include <splat/cpu_rasteriser.hpp>
#include <splat/ipu_rasteriser.hpp>
#include <splat/file_io.hpp>
#include <splat/serialise.hpp>

#include <splat/ipu_geometry.hpp>

#include <remote_ui/InterfaceServer.hpp>
#include <remote_ui/AsyncTask.hpp>

#include <pvti/pvti.hpp>

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
   "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.")
  ("ui-port", po::value<int>()->default_value(0), "Start a remote user-interface server on the specified port.")
  ("device", po::value<std::string>()->default_value("cpu"),
   "Choose the render device")
  ("no-amp", po::bool_switch()->default_value(true),
   "Disable use of optimised AMP codelets.")
  ("flip-up", po::bool_switch()->default_value(false),
   "Flip the world up-axis. Use this for COLMAP / Gaussian Splatting SLAM scenes "
   "(where world +Y points down) so the scene renders right-side-up.")
  ("flip-scene", po::bool_switch()->default_value(false),
   "Rotate the scene 180 deg around the world X axis (negates Y and Z of all "
   "world coordinates). Use this for scenes that appear both upside-down AND "
   "facing away from the camera.")
  ("paired-shots-dir", po::value<std::string>()->default_value("paired_shots"),
   "Where to save framebuffer + pose JSON when the client clicks Screenshot. "
   "A sibling watcher (tools/gpu_watch.py) turns each JSON into a GPU reference render.")
  ("from-pose", po::value<std::string>()->default_value(""),
   "Path to a sidecar .json saved by the Screenshot button. Loads the view "
   "matrix and FOV from it so the server starts with that exact pose.")
  ("play-path", po::value<std::string>()->default_value(""),
   "Path to a .traj file (see tools/sample_orbit_path.py). When set, the "
   "server overrides the camera pose each rendered frame with the next pose "
   "from the trajectory and loops at the end. Client WASD/mouse-look pose "
   "input is ignored while playback is active (FOV and Stop still work).")
  ("play", po::bool_switch()->default_value(false),
   "Auto-find a trajectory matching --input. Looks for "
   "tools/benchmark_traj_<stem>.traj (or ../tools/...) where <stem> is the "
   "PLY filename without extension. Equivalent to --play-path with that path.")
  ("benchmark", po::value<int>()->default_value(0),
   "Run N frames headlessly with a fixed pose and report mean FPS, then exit. "
   "No --ui-port needed. Uses initial view or --from-pose if provided.")
  ("bench-static", po::value<int>()->default_value(0),
   "Run N frames headlessly at a single fixed pose (from --from-pose or the "
   "default initial view) and write per-frame timing to benchmark_profile.csv "
   "in the same schema as the trajectory benchmark. Use this to compare "
   "fixed-pose throughput against orbit (--play) throughput.")
  ("bench-no-readback", po::bool_switch()->default_value(false),
   "Skip device->host readbacks (visible counts + per-tile phase cycles) "
   "every frame during a trajectory or static benchmark. The last frame's "
   "framebuffer is still read once at the end. Use this to measure pure "
   "on-chip compute power, excluding the I/O bandwidth contribution. "
   "Per-frame visible/phase columns in the CSV will be 0 in this mode.")
  ("bench-readback-frames", po::bool_switch()->default_value(false),
   "Also read the full framebuffer (3.6 MB at 1280x720 RGBX) every frame "
   "during a trajectory or static benchmark. Models the real interactive "
   "cost where the host needs the pixels each frame; for comparison against "
   "--bench-no-readback / default (counts+cycles only) the difference in "
   "power_W and total_ms reveals the framebuffer-readback contribution.")
  ("device-loop", po::bool_switch()->default_value(false),
   "Use RepeatWhileTrue device-side loop for IPU rendering. Eliminates "
   "per-frame host-device barrier for higher throughput.")
  ("orbit", po::value<float>()->default_value(0.f),
   "Auto-orbit: rotate yaw by this many degrees per frame (e.g. --orbit 1.0 "
   "for a slow turntable). Overrides client yaw control.")
  ("gather-mode", po::value<std::string>()->default_value("news"),
   "Render path: 'news' (on-chip Manhattan routing, default) or 'multislice' "
   "(host-assisted discovery + popops::multiSlice gather; faster, flicker-free, "
   "discovery on host). Experimental — branch jdl-experiment.")
  ("gather-cap", po::value<int>()->default_value(400),
   "Per-tile Gaussian capacity for --gather-mode multislice. Lower = faster but "
   "drops the farthest Gaussians on dense tiles (e.g. 200).")
  ("device-discovery", po::bool_switch()->default_value(false),
   "Run Gaussian discovery (projection + tile assignment) on the IPU instead "
   "of the host CPU. Only applies to --gather-mode multislice.");
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Points& pts, splat::TiledFramebuffer& fb, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts, fb, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Gaussians& pts, splat::TiledFramebuffer& fb, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts, fb, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
}

int main(int argc, char** argv) {
  pvti::TraceChannel traceChannel = {"splatter"};

  boost::program_options::options_description desc;
  addOptions(desc);
  boost::program_options::variables_map args;
  try {
    args = parseOptions(argc, argv, desc);
    setupLogging(args);
  } catch (const std::exception& e) {
    ipu_utils::logger()->info("Exiting after: {}.", e.what());
    return EXIT_FAILURE;
  }

   // Create an instance of the Ply class to store the gaussian properties
  splat::Ply ply;

  auto xyzFile = args["input"].as<std::string>();
  auto pts = splat::loadPoints(xyzFile, ply);
  splat::Bounds3f bb(pts);

  ipu_utils::logger()->info("Total point count: {}", pts.size());
  ipu_utils::logger()->info("Point bounds (world space): {}", bb);

  // Translate all points so the centroid is at the origin:
  {
    const auto bbCentre = bb.centroid();
    for (auto& v : pts) {
      v.p -= bbCentre;
    }
    bb = splat::Bounds3f(pts);
  }

  // bb.max = {1.f, 1.f, 1.f};
  // bb.min = {-1.f, -1.f, -1.f};
  // Splat all the points into an OpenCV image:
  auto imagePtr = std::make_unique<cv::Mat>(720, 1280, CV_8UC3);
  auto imagePtrBuffered = std::make_unique<cv::Mat>(imagePtr->rows, imagePtr->cols, CV_8UC3);
  const float aspect = imagePtr->cols / (float)imagePtr->rows;

  //Bb size
  ipu_utils::logger()->info("BB size: {}", bb.diagonal().length());


  // Construct some tiled framebuffer histograms:
  splat::TiledFramebuffer fb(imagePtr->cols, imagePtr->rows, IPU_TILEWIDTH, IPU_TILEHEIGHT);
  auto pointCounts = std::vector<std::uint32_t>(fb.numTiles, 0u);

  auto num_pixels = imagePtr->rows * imagePtr->cols;
  auto pixels_per_tile = num_pixels / fb.numTiles;
  ipu_utils::logger()->info("Number of pixels in framebuffer: {}", num_pixels);
  ipu_utils::logger()->info("Number of tiles in framebuffer: {}", fb.numTiles);
  ipu_utils::logger()->info("Number of pixels per tile: {}", pixels_per_tile);

  float x = 719.f;
  float y = 1279.f;
  auto tileId = fb.pixCoordToTile(x, y);
  ipu_utils::logger()->info("Tile index test. Pix coord {}, {} -> tile id: {}", x, y, tileId);


  auto centre = bb.centroid();
  // make fb.numTiles copies of a 2D gaussian
  splat::Gaussians gsns;
  ipu_utils::logger()->info("Generating {} gaussians", pts.size());


  // (/ 1.0 (* 2.0 (sqrt pi)))
  const float SH_C0 = 0.28209479177387814f;
  
  for (std::size_t i = 0; i < pts.size(); i++) {
    auto pt = pts[i].p;
    splat::Gaussian3D g;
    g.mean = {pt.x, pt.y, pt.z};
    if (ply.f_dc[0].values.size() > 0) {
      glm::vec3 colour = {SH_C0 * ply.f_dc[0].values[i],
                      SH_C0 * ply.f_dc[1].values[i],
                      SH_C0 * ply.f_dc[2].values[i]};
      colour += 0.5f;
      colour = glm::max(colour, glm::vec3(0.f));
      float sigmoid_opacity = 1.0f / (1.0f + expf(-ply.opacity.values[i]));
      g.colour = {colour.x, colour.y, colour.z, sigmoid_opacity};
      g.scale = {ply.scale[0].values[i], ply.scale[1].values[i], ply.scale[2].values[i]};
      // g.scale = {-5.f, -5.f, -5.f};
      g.rot = {ply.rot[0].values[i], ply.rot[1].values[i], ply.rot[2].values[i], ply.rot[3].values[i]};

      // printf("scale: %f %f %f\n", g.scale.x, g.scale.y, g.scale.z);
      // printf("rot: %f %f %f %f\n", g.rot.x, g.rot.y, g.rot.z, g.rot.w);
      // printf("colour: %f %f %f %f\n", g.colour.x, g.colour.y, g.colour.z, g.colour.w);
      // printf("mean: %f %f %f %f\n", g.mean.x, g.mean.y, g.mean.z, g.mean.w);
    } else {
      g.colour = {0.05f, 0.05f, 0.05f, 1.0f};
      g.scale = {1.f, 1.f, 1.f};
    }
    g.gid = static_cast<float>(i) + 1.0f;
    gsns.push_back(g);
  }


  auto ipuSplatter = createIpuBuilder(gsns, fb, args["no-amp"].as<bool>());
  const bool useGather = args["gather-mode"].as<std::string>() == "multislice";
  if (useGather) {
    const int cap = args["gather-cap"].as<int>();
    const bool devDisc = args["device-discovery"].as<bool>();
    ipu_utils::logger()->info("Render path: multiSlice gather ({}discovery), cap {}",
                              devDisc ? "device " : "host ", cap);
    ipuSplatter->setGatherCap((unsigned)cap);
    ipuSplatter->setDeviceDiscovery(devDisc);
  }
  ipuSplatter->setGatherMode(useGather);
  ipu_utils::GraphManager gm;
  gm.compileOrLoad(*ipuSplatter);

  // Setup a user interface server if requested:
  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;

  state.device = args.at("device").as<std::string>();

  // --from-pose: load a previously-saved Screenshot sidecar and use it as the
  // starting camera. The JSON stores the final COLMAP-convention dynamicView;
  // the server still applies its OpenGL->COLMAP flip at render time, so we
  // flip again here (self-inverse) to get the OpenGL matrix the pipeline
  // expects as input. Also loads fov.
  std::vector<float> loadedViewColmap;   // empty unless --from-pose set
  float loadedFovHalfRad = 0.f;
  {
    const std::string posePath = args["from-pose"].as<std::string>();
    if (!posePath.empty()) {
      std::ifstream f(posePath);
      if (!f) {
        ipu_utils::logger()->warn("Could not open --from-pose {}; ignoring", posePath);
      } else {
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        auto findKey = [&](const std::string& key) { return content.find("\"" + key + "\""); };

        // view_matrix: 16 floats inside [ ... ]
        auto k = findKey("view_matrix");
        if (k != std::string::npos) {
          auto lb = content.find('[', k);
          auto rb = content.find(']', lb);
          if (lb != std::string::npos && rb != std::string::npos) {
            std::string body = content.substr(lb + 1, rb - lb - 1);
            std::replace(body.begin(), body.end(), ',', ' ');
            std::stringstream ss(body);
            float v;
            while (ss >> v) loadedViewColmap.push_back(v);
          }
        }
        if (loadedViewColmap.size() != 16) {
          ipu_utils::logger()->warn("--from-pose {}: view_matrix must have 16 floats, got {}",
                                    posePath, loadedViewColmap.size());
          loadedViewColmap.clear();
        }

        // fov_half_rad
        auto kf = findKey("fov_half_rad");
        if (kf != std::string::npos) {
          auto colon = content.find(':', kf);
          try { loadedFovHalfRad = std::stof(content.substr(colon + 1)); }
          catch (...) { loadedFovHalfRad = 0.f; }
        }
        if (loadedFovHalfRad > 0.f) state.fov = loadedFovHalfRad;
        ipu_utils::logger()->info("--from-pose loaded from {} (fov_half_rad = {})",
                                  posePath, loadedFovHalfRad);
      }
    }
  }

  // --play-path: load a flat .traj file (see tools/sample_orbit_path.py) and
  // play it back, one pose per rendered frame, looping at the end. Trajectory
  // poses are stored in COLMAP convention (same as --from-pose), so the
  // playback override skips the OpenGL->COLMAP flip the interactive path uses.
  // Format: "fov_half_rad: <f>", "ply: <p>" headers, then lines beginning
  // with "v <16 floats>" — one view matrix per frame in GLM column-major.
  std::vector<float> playbackPoses;   // 16 floats per pose, flat
  {
    std::string playPath = args["play-path"].as<std::string>();
    if (playPath.empty() && args["play"].as<bool>()) {
      // Derive from --input: e.g. "../data/sloth.ply" -> stem "sloth".
      // Try a few candidate locations so this works whether the binary runs
      // from build/, repo root, or elsewhere.
      const std::string stem = std::filesystem::path(xyzFile).stem().string();
      const std::vector<std::string> candidates = {
        "tools/benchmark_traj_" + stem + ".traj",
        "../tools/benchmark_traj_" + stem + ".traj",
      };
      for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) { playPath = c; break; }
      }
      if (playPath.empty()) {
        ipu_utils::logger()->warn("--play set but no trajectory found for stem '{}' "
                                  "(tried {} candidates); playback disabled",
                                  stem, candidates.size());
      } else {
        ipu_utils::logger()->info("--play auto-resolved to {}", playPath);
      }
    }
    if (!playPath.empty()) {
      std::ifstream f(playPath);
      if (!f) {
        ipu_utils::logger()->warn("Could not open --play-path {}; ignoring", playPath);
      } else {
        std::string line;
        while (std::getline(f, line)) {
          size_t s = line.find_first_not_of(" \t");
          if (s == std::string::npos || line[s] == '#') continue;
          if (line.compare(s, 13, "fov_half_rad:") == 0) {
            try { loadedFovHalfRad = std::stof(line.substr(s + 13)); }
            catch (...) {}
          } else if (line[s] == 'v' &&
                     (s + 1 == line.size() || std::isspace((unsigned char)line[s + 1]))) {
            std::stringstream ss(line.substr(s + 1));
            float v;
            int n = 0;
            float buf[16];
            while (n < 16 && ss >> v) buf[n++] = v;
            if (n == 16) {
              for (int i = 0; i < 16; ++i) playbackPoses.push_back(buf[i]);
            }
          }
        }
        const size_t numPoses = playbackPoses.size() / 16;
        if (loadedFovHalfRad > 0.f) state.fov = loadedFovHalfRad;
        ipu_utils::logger()->info("--play-path loaded {} poses from {} (fov_half_rad = {})",
                                  numPoses, playPath, loadedFovHalfRad);
        if (numPoses == 0) {
          ipu_utils::logger()->warn("--play-path {} contained no 'v <16 floats>' lines; "
                                    "playback disabled", playPath);
          playbackPoses.clear();
        }
      }
    }
  }

  auto uiPort = args.at("ui-port").as<int>();
  if (uiPort) {
    uiServer.reset(new InterfaceServer(uiPort));
    if (loadedFovHalfRad > 0.f) uiServer->setInitialFov(loadedFovHalfRad);
    uiServer->start();
    uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    uiServer->updateFov(state.fov);
  }

  // Set up the modelling and projection transforms in an OpenGL compatible way.
  // For COLMAP-derived scenes (e.g. Gaussian Splatting SLAM outputs), world +Y
  // points DOWN — use --flip-up to get a right-side-up render.
  const bool flipUp = args["flip-up"].as<bool>();
  glm::vec3 upAxis = flipUp ? glm::vec3(0.f, -1.f, 0.f) : glm::vec3(0.f, 1.f, 0.f);
  auto viewMatrix = splat::lookAtBoundingBox(bb, upAxis, 2.f);
  ipu_utils::logger()->info("Using world up = {}", flipUp ? "-Y (COLMAP/SLAM)" : "+Y (OpenGL)");

  // --flip-scene: rotate world 180 deg around X (negate Y,Z of world coords).
  // Equivalent to post-multiplying the view matrix by diag(1,-1,-1,1). Fixes
  // scenes that load both upside-down and facing away from the camera.
  if (args["flip-scene"].as<bool>()) {
    glm::mat4 R_x180(1.0f);
    R_x180[1][1] = -1.0f;
    R_x180[2][2] = -1.0f;
    viewMatrix = viewMatrix * R_x180;
    ipu_utils::logger()->info("Scene rotated 180 deg around X (--flip-scene)");
  }

  // If --from-pose loaded a view matrix, use it in place of lookAtBoundingBox.
  // The JSON holds the COLMAP-convention dynamicView; the pipeline later
  // re-applies kOpenGLToColmap, so we flip here first (self-inverse) to get
  // back the OpenGL matrix the pipeline expects as input.
  if (loadedViewColmap.size() == 16) {
    glm::mat4 V_colmap(0.f);
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
        V_colmap[c][r] = loadedViewColmap[c * 4 + r];
    glm::mat4 flipYZ(1.0f);
    flipYZ[1][1] = -1.0f;
    flipYZ[2][2] = -1.0f;
    viewMatrix = flipYZ * V_colmap;
    ipu_utils::logger()->info("Initial view matrix overridden by --from-pose");
  }

  // Transform the BB to camera/eye space:
  splat::Bounds3f bbInCamera(
    viewMatrix * glm::vec4(bb.min, 1.f),
    viewMatrix * glm::vec4(bb.max, 1.f)
  );

  ipu_utils::logger()->info("Point bounds (eye space): {}", bbInCamera);
  auto projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);

  ipuSplatter->updateModelView(viewMatrix);
  ipuSplatter->updateProjection(projection);
  gm.prepareEngine();

  // --bench-static N: when set without --play, "fake" the trajectory bench by
  // populating playbackPoses with a single pose = the current viewMatrix
  // converted to COLMAP convention. The trajectory benchmark below then runs
  // N full loops through a 1-pose trajectory = N frames at the fixed view.
  // This shares all the CSV/printing code with the orbit bench so static and
  // orbit results are directly comparable.
  const int benchStatic = args["bench-static"].as<int>();
  int benchmarkSubsteps = args["benchmark"].as<int>();
  if (benchStatic > 0 && playbackPoses.empty()) {
    static const glm::mat4 kFlipStatic = glm::mat4(
        glm::vec4( 1.f,  0.f,  0.f, 0.f),
        glm::vec4( 0.f, -1.f,  0.f, 0.f),
        glm::vec4( 0.f,  0.f, -1.f, 0.f),
        glm::vec4( 0.f,  0.f,  0.f, 1.f));
    glm::mat4 staticView = kFlipStatic * viewMatrix;
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
        playbackPoses.push_back(staticView[c][r]);
    benchmarkSubsteps = benchStatic;
    ipu_utils::logger()->info("--bench-static {}: running {} frames at the fixed initial pose",
                              benchStatic, benchStatic);
  }

  // --benchmark N: run N substeps per zoom level with per-phase timing and
  // convergence tracking. Outputs CSV with route/blend/exchange breakdown
  // and total visible Gaussian count per substep.
  //
  // When --play-path / --play is also set, the behaviour changes: N becomes
  // the number of full trajectory loops (spins), and each frame uses the
  // next pose from the trajectory. CSV gets one row per frame.
  if (benchmarkSubsteps > 0) {
    static const glm::mat4 kFlip = glm::mat4(
        glm::vec4( 1.f,  0.f,  0.f, 0.f),
        glm::vec4( 0.f, -1.f,  0.f, 0.f),
        glm::vec4( 0.f,  0.f, -1.f, 0.f),
        glm::vec4( 0.f,  0.f,  0.f, 1.f));

    // Initialize: first execute connects streams and writes vertices
    gm.execute(*ipuSplatter);

    // ---- multiSlice gather benchmark (Stage 2) ----
    // The gather path doesn't build the NEWS single_route/blend/exchange
    // programs, so it gets its own loop: one gm.execute (= executeGather) per
    // pose, recording the host discovery / stream / device run / readback split.
    if (useGather) {
      if (playbackPoses.empty()) {
        ipu_utils::logger()->error("--gather-mode multislice benchmark needs --bench-static or --play-path");
        return EXIT_FAILURE;
      }
      const size_t numPoses = playbackPoses.size() / 16;
      const size_t totalFrames = static_cast<size_t>(benchmarkSubsteps) * numPoses;

      FILE* csv = fopen("benchmark_profile.csv", "w");
      fprintf(csv, "frame,pose_idx,discovery_ms,stream_ms,mvp_ms,gather_ms,project_ms,blend_ms,readback_ms,total_ms,assigned\n");
      printf("Gather benchmark: %zu frames (%d x %zu poses)\n", totalFrames, benchmarkSubsteps, numPoses);

      using clk = std::chrono::steady_clock;
      auto bench_start = clk::now();
      for (size_t f = 0; f < totalFrames; ++f) {
        const float* vm = &playbackPoses[16 * (f % numPoses)];
        glm::mat4 V_colmap(0.f);
        for (int c = 0; c < 4; ++c)
          for (int r = 0; r < 4; ++r)
            V_colmap[c][r] = vm[c * 4 + r];
        ipuSplatter->updateModelView(V_colmap);
        ipuSplatter->updateProjection(projection);
        ipuSplatter->updateFocalLengths(state.fov, 0.f);

        gm.execute(*ipuSplatter);  // executeGather: discovery + stream + run + readback
        auto gt = ipuSplatter->getLastGatherTiming();
        fprintf(csv, "%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%u\n",
                f, f % numPoses, gt.discovery_ms, gt.stream_ms, gt.mvp_ms,
                gt.gather_ms, gt.project_ms, gt.blend_ms,
                gt.readback_ms, gt.total_ms(), gt.assigned);
        if (f % 60 == 0 || f + 1 == totalFrames) {
          printf("  frame %5zu  disc %.1f  stream %.1f  mvp %.1f  gather %.1f  "
                 "proj %.1f  blend %.1f  rb %.1f  total %.1f ms  assigned %u\n",
                 f + 1, gt.discovery_ms, gt.stream_ms, gt.mvp_ms, gt.gather_ms,
                 gt.project_ms, gt.blend_ms, gt.readback_ms,
                 gt.total_ms(), gt.assigned);
        }
      }
      auto bench_end = clk::now();
      double bench_secs = std::chrono::duration<double>(bench_end - bench_start).count();
      fclose(csv);

      ipuSplatter->getFrameBuffer(*imagePtr);
      cv::imwrite("benchmark_last_frame.png", *imagePtr);
      printf("Gather benchmark complete: %zu frames in %.2fs (%.1f FPS).\n",
             totalFrames, bench_secs, totalFrames / bench_secs);
      printf("Per-frame timing saved to benchmark_profile.csv\n");
      return EXIT_SUCCESS;
    }

    // Trajectory-driven benchmark: iterate `benchmarkSubsteps` full loops
    // through the loaded trajectory, one render per pose.
    if (!playbackPoses.empty()) {
      const size_t numPoses = playbackPoses.size() / 16;
      const size_t totalFrames = static_cast<size_t>(benchmarkSubsteps) * numPoses;

      FILE* csv = fopen("benchmark_profile.csv", "w");
      fprintf(csv, "frame,pose_idx,wall_ms,route_ms,blend_ms,exchange_ms,total_ms,mvp_ms,"
                   "clear_min,clear_mean,clear_max,"
                   "routing_min,routing_mean,routing_max,"
                   "proj_min,proj_mean,proj_max,"
                   "sort_min,sort_mean,sort_max,"
                   "total_cyc_min,total_cyc_mean,total_cyc_max,"
                   "total_visible\n");

      printf("Trajectory benchmark: %zu frames (%d spins x %zu poses)\n",
             totalFrames, benchmarkSubsteps, numPoses);
      printf("\n%-7s %4s %9s %9s %9s | %-26s | %-26s | %-26s | %10s\n",
             "Frame", "Pose", "Route", "Blend", "Total",
             "  Routing min/mean/max", "  Project min/mean/max", "     Sort min/mean/max", "Visible");
      printf("%s\n", std::string(140, '-').c_str());

      using clk = std::chrono::steady_clock;
      auto bench_start = clk::now();

      for (size_t f = 0; f < totalFrames; ++f) {
        const float* vm = &playbackPoses[16 * (f % numPoses)];
        glm::mat4 V_colmap(0.f);
        for (int c = 0; c < 4; ++c)
          for (int r = 0; r < 4; ++r)
            V_colmap[c][r] = vm[c * 4 + r];

        // Trajectory poses are already in COLMAP convention -> feed directly.
        ipuSplatter->updateModelView(V_colmap);
        ipuSplatter->updateProjection(projection);
        ipuSplatter->updateFocalLengths(state.fov, 0.f);

        auto t0 = clk::now();
        ipuSplatter->broadcastMVP();
        auto t1 = clk::now();
        double mvp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const bool noReadback = args["bench-no-readback"].as<bool>();
        const bool readbackFrames = args["bench-readback-frames"].as<bool>();

        auto wallStart = clk::now();
        auto timing = ipuSplatter->runSingleSubstep();
        if (!noReadback) {
          ipuSplatter->readbackCounts();
          ipuSplatter->readbackPhaseCycles();
        }
        if (readbackFrames) {
          ipuSplatter->readbackFramebuffer();
        }
        auto wallEnd = clk::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();

        unsigned totalVisible = noReadback ? 0u : ipuSplatter->getTotalSplatCount();
        splat::IpuSplatter::CycleBreakdown bd{};
        if (!noReadback) bd = ipuSplatter->getPhaseCycleStats();

        fprintf(csv, "%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%u\n",
                f, f % numPoses, wall_ms,
                timing.route_ms, timing.blend_ms, timing.exchange_ms,
                timing.compute_ms, mvp_ms,
                bd.clear.min_ms, bd.clear.mean_ms, bd.clear.max_ms,
                bd.routing.min_ms, bd.routing.mean_ms, bd.routing.max_ms,
                bd.projection.min_ms, bd.projection.mean_ms, bd.projection.max_ms,
                bd.sorting.min_ms, bd.sorting.mean_ms, bd.sorting.max_ms,
                bd.total.min_ms, bd.total.mean_ms, bd.total.max_ms,
                totalVisible);

        if (f % 60 == 0 || f + 1 == totalFrames) {
          printf("%-7zu %4zu %9.2f %9.2f %9.2f | %7.2f/%7.2f/%7.2f | %7.2f/%7.2f/%7.2f | %7.3f/%7.3f/%7.3f | %10u\n",
                 f + 1, f % numPoses,
                 timing.route_ms, timing.blend_ms, timing.compute_ms,
                 bd.routing.min_ms, bd.routing.mean_ms, bd.routing.max_ms,
                 bd.projection.min_ms, bd.projection.mean_ms, bd.projection.max_ms,
                 bd.sorting.min_ms, bd.sorting.mean_ms, bd.sorting.max_ms,
                 totalVisible);
        }
      }

      auto bench_end = clk::now();
      double bench_secs = std::chrono::duration<double>(bench_end - bench_start).count();
      fclose(csv);

      // Save the last rendered frame so the wrapper can verify the scene looked right.
      ipuSplatter->readbackFramebuffer();
      ipuSplatter->getFrameBuffer(*imagePtr);
      cv::imwrite("benchmark_last_frame.png", *imagePtr);

      printf("Trajectory benchmark complete: %zu frames in %.2fs (%.1f FPS).\n",
             totalFrames, bench_secs, totalFrames / bench_secs);
      printf("Per-frame timing saved to benchmark_profile.csv\n");
      printf("Last frame saved to benchmark_last_frame.png\n");
      return EXIT_SUCCESS;
    }

    const float zoomLevels[] = {1.0f, 1.2f, 1.5f, 2.0f};
    const int nZooms = sizeof(zoomLevels) / sizeof(zoomLevels[0]);
    const float sceneRadius = glm::length(bb.diagonal()) * 0.5f;

    FILE* csv = fopen("benchmark_profile.csv", "w");
    fprintf(csv, "zoom,substep,route_ms,blend_ms,exchange_ms,total_ms,mvp_ms,"
                 "clear_min,clear_mean,clear_max,"
                 "routing_min,routing_mean,routing_max,"
                 "proj_min,proj_mean,proj_max,"
                 "sort_min,sort_mean,sort_max,"
                 "total_cyc_min,total_cyc_mean,total_cyc_max,"
                 "total_visible\n");

    printf("\n%-5s %4s %9s %9s %9s | %-26s | %-26s | %-26s | %10s\n",
           "Zoom", "Step", "Route", "Blend", "Total",
           "  Routing min/mean/max", "  Project min/mean/max", "     Sort min/mean/max", "Visible");
    printf("%s\n", std::string(140, '-').c_str());

    for (int z = 0; z < nZooms; ++z) {
      float moveBack = sceneRadius * 0.5f * (zoomLevels[z] - 1.0f);
      glm::mat4 zoomTranslate = glm::translate(glm::mat4(1.0f),
                                                glm::vec3(0.f, 0.f, -moveBack));
      auto benchView = kFlip * zoomTranslate * viewMatrix;

      ipuSplatter->updateModelView(benchView);
      ipuSplatter->updateProjection(projection);
      ipuSplatter->updateFocalLengths(state.fov, 0.f);

      using clk = std::chrono::steady_clock;
      auto t0 = clk::now();
      ipuSplatter->broadcastMVP();
      auto t1 = clk::now();
      double mvp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      unsigned prevCount = 0;
      int settledAt = -1;

      for (int s = 0; s < benchmarkSubsteps; ++s) {
        auto timing = ipuSplatter->runSingleSubstep();
        ipuSplatter->readbackCounts();
        ipuSplatter->readbackPhaseCycles();
        unsigned totalVisible = ipuSplatter->getTotalSplatCount();

        auto bd = ipuSplatter->getPhaseCycleStats();

        fprintf(csv, "%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%.4f,%.4f,%.4f,"
                     "%u\n",
                zoomLevels[z], s,
                timing.route_ms, timing.blend_ms, timing.exchange_ms,
                timing.compute_ms, (s == 0 ? mvp_ms : 0.0),
                bd.clear.min_ms, bd.clear.mean_ms, bd.clear.max_ms,
                bd.routing.min_ms, bd.routing.mean_ms, bd.routing.max_ms,
                bd.projection.min_ms, bd.projection.mean_ms, bd.projection.max_ms,
                bd.sorting.min_ms, bd.sorting.mean_ms, bd.sorting.max_ms,
                bd.total.min_ms, bd.total.mean_ms, bd.total.max_ms,
                totalVisible);

        if (s % 5 == 0 || s < 5) {
          printf("%-5.2f %4d %9.2f %9.2f %9.2f | %7.2f/%7.2f/%7.2f | %7.2f/%7.2f/%7.2f | %7.3f/%7.3f/%7.3f | %10u\n",
                 zoomLevels[z], s,
                 timing.route_ms, timing.blend_ms, timing.compute_ms,
                 bd.routing.min_ms, bd.routing.mean_ms, bd.routing.max_ms,
                 bd.projection.min_ms, bd.projection.mean_ms, bd.projection.max_ms,
                 bd.sorting.min_ms, bd.sorting.mean_ms, bd.sorting.max_ms,
                 totalVisible);
        }

        if (settledAt < 0 && s > 0 && prevCount > 0) {
          double delta = std::abs((double)totalVisible - (double)prevCount) / prevCount;
          if (delta < 0.001) settledAt = s;
        }
        prevCount = totalVisible;
      }

      ipuSplatter->readbackFramebuffer();
      ipuSplatter->getFrameBuffer(*imagePtr);
      char fname[64];
      snprintf(fname, sizeof(fname), "benchmark_zoom_%.2f.png", zoomLevels[z]);
      cv::imwrite(fname, *imagePtr);

      printf("  -> zoom %.2f settled at substep %d, %u visible Gaussians\n\n",
             zoomLevels[z], settledAt, prevCount);
    }

    fclose(csv);
    printf("Per-phase timing saved to benchmark_profile.csv\n");
    printf("Saved %d frames as benchmark_zoom_*.png\n", nZooms);
    return EXIT_SUCCESS;
  }

  std::vector<glm::vec4> clipSpace;
  clipSpace.reserve(pts.size());
  splat::TiledFramebuffer cpufb(CPU_TILEWIDTH, CPU_TILEHEIGHT);
  splat::Viewport vp(0.f, 0.f, IMWIDTH, IMHEIGHT);

  // Video is encoded and sent in a separate thread:
  AsyncTask hostProcessing;
  auto uiUpdateFunc = [&]() {
    {
      pvti::Tracepoint scoped(&traceChannel, "ui_update");
      uiServer->sendHistogram(pointCounts);
      uiServer->sendPreviewImage(*imagePtrBuffered);
    }
    if (state.device == "cpu") {
      {
        pvti::Tracepoint scope(&traceChannel, "build_histogram");
        splat::buildTileHistogram(pointCounts, clipSpace, cpufb, vp);
      }
    } else {
      {
        pvti::Tracepoint scope(&traceChannel, "build_histogram");
        ipuSplatter->getIPUHistogram(pointCounts);
      }
    }
  };

  auto secondsElapsed = 0.0;

  const bool useDeviceLoop = args["device-loop"].as<bool>() && state.device == "ipu";
  if (useDeviceLoop) {
    ipuSplatter->setDeviceLoopMode(true);
    ipu_utils::logger()->info("Device loop mode enabled (RepeatWhileTrue)");
  }

  auto  dynamicView = viewMatrix;
  float orbitYawAccum = 0.f;
  float orbitPlusPhase = 0.f;  // sine phase for the plus-sign (cross) sweep
  bool  prevOrbitActive = false;
  bool  prevOrbitPlus = false;
  bool  prevOrbitMinus = false;
  bool  loopTracking = false;
  float loopTargetPhase = 0.f;
  size_t playbackFrameIdx = 0;
  do {
    auto startTime = std::chrono::steady_clock::now();
    *imagePtr = 0;
    std::uint32_t count = 0u;

    // --play-path: override the camera pose with the next trajectory pose
    // (already in COLMAP convention) and loop at the end. Runs at the
    // server's render FPS — more frames in the trajectory = slower playback.
    if (!playbackPoses.empty()) {
      const size_t numPoses = playbackPoses.size() / 16;
      const float* vm = &playbackPoses[16 * (playbackFrameIdx % numPoses)];
      glm::mat4 V_colmap(0.f);
      for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
          V_colmap[c][r] = vm[c * 4 + r];
      dynamicView = V_colmap;
      ++playbackFrameIdx;
    }

    if (state.device == "cpu") {
      pvti::Tracepoint scoped(&traceChannel, "mvp_transform_cpu");
      projectPoints(pts, projection, dynamicView, clipSpace);
      {
        pvti::Tracepoint scope(&traceChannel, "splatting_cpu");
        count = splat::splatPoints(*imagePtr, clipSpace, pts, projection, dynamicView, cpufb, vp);
      }
    } else if (state.device == "ipu") {
      pvti::Tracepoint scoped(&traceChannel, "mvp_transform_ipu");
      ipuSplatter->updateModelView(dynamicView);
      ipuSplatter->updateProjection(projection);

      ipuSplatter->updateFocalLengths(state.fov, state.lambda1);
      gm.execute(*ipuSplatter);
      ipuSplatter->getFrameBuffer(*imagePtr);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto splatTimeSecs = std::chrono::duration<double>(endTime - startTime).count();

    // Send the pure render time (pre-frame-upload) to the client every frame.
    if (uiServer) {
      uiServer->sendRenderTime(float(splatTimeSecs * 1000.0));
    }

    // Handle paired-screenshot request: save the current framebuffer and a JSON
    // sidecar describing the pose. tools/gpu_watch.py picks the JSON up and
    // runs diff-gaussian-rasterization at the same pose to produce the matching
    // GPU reference image.
    if (uiServer && uiServer->consumeScreenshotRequest()) {
      namespace fs = std::filesystem;
      fs::path outDir = args["paired-shots-dir"].as<std::string>();
      std::error_code ec;
      fs::create_directories(outDir, ec);

      auto now = std::time(nullptr);
      char ts[32];
      std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&now));
      std::string stem = std::string("screenshot-") + ts;
      fs::path pngPath  = outDir / (stem + ".png");
      fs::path jsonPath = outDir / (stem + ".json");

      cv::imwrite(pngPath.string(), *imagePtr);

      std::ofstream js(jsonPath);
      js << "{\n";
      js << "  \"view_matrix\": [";
      for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
          js << dynamicView[c][r];
          if (!(c == 3 && r == 3)) js << ", ";
        }
      }
      js << "],\n";
      js << "  \"fov_half_rad\": " << state.fov << ",\n";
      // Always store an absolute path so the watcher (which runs from a
      // different cwd) can find the PLY regardless.
      std::error_code ecAbs;
      auto absPly = std::filesystem::absolute(xyzFile, ecAbs);
      js << "  \"ply\": \"" << (ecAbs ? xyzFile : absPly.string()) << "\"\n";
      js << "}\n";
      ipu_utils::logger()->info("Saved paired-shot to {} (+ .json)", pngPath.string());
    }

    secondsElapsed += splatTimeSecs;
    if (secondsElapsed > 3.f) {
      ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pts.size()/splatTimeSecs);
      // print dynamic view matrix:

      // for (int i = 0; i < 4; i++) {
      //   ipu_utils::logger()->info("Dynamic view matrix: {} {} {} {}", dynamicView[i][0], dynamicView[i][1], dynamicView[i][2], dynamicView[i][3]);
      // }
    }

    if (uiServer) {
      hostProcessing.waitForCompletion();
      std::swap(imagePtr, imagePtrBuffered);
      hostProcessing.run(uiUpdateFunc);

      state = uiServer->consumeState();
      // Update projection:
      projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);
      // Update modelview:
      if (secondsElapsed >= 3.f) {
        // Print the current dynamicView — this is what the IPU actually renders
        // with and what a reference GPU renderer needs to match the frame.
        // GLM is column-major: dynamicView[i] is the i-th column, so each line
        // logs one column. Feed these 16 numbers to the GPU script as
        //   --view-matrix "<col0.x col0.y col0.z col0.w   col1... col2... col3...>"
        for (int i = 0; i < 4; i++) {
          ipu_utils::logger()->info("Dynamic view matrix: {} {} {} {}", dynamicView[i][0], dynamicView[i][1], dynamicView[i][2], dynamicView[i][3]);
        }

        //print state
        printf("X: %f\n", state.X);
        printf("Y: %f\n", state.Y);
        printf("Z: %f\n", state.Z);

        
        printf("envRotationDegrees: %f\n", state.envRotationDegrees);
        printf("envRotationDegrees2: %f\n", state.envRotationDegrees2);
        printf("lambda1: %f\n", state.lambda1);
        printf("fov: %f\n", state.fov);
        secondsElapsed = 0.0;

      }

//       envRotationDegrees: 96.654427
// envRotationDegrees2: 2.726311
// fov: 0.352075

// envRotationDegrees: 85.763603
// envRotationDegrees2: 184.763657
// fov: 0.433323
      // FPS camera control (WASD + mouse-look from the client):
      //   envRotationDegrees   = pitch (rotation about camera X)
      //   envRotationDegrees2  = yaw   (rotation about world Y)
      //   (X, Y, Z)            = camera offset, in units of the scene diagonal
      //
      // Final view = R_pitch * R_yaw * T(-offset * sceneScale) * initialView
      //
      // Scaling the client's offset by the scene diagonal makes WASD motion
      // feel the same regardless of scene units (COLMAP scenes can be tiny or
      // large). With all params zero, dynamicView == viewMatrix.
      const float orbitStepCLI = args["orbit"].as<float>();
      bool orbiting = orbitStepCLI != 0.f || state.orbitActive || state.orbitPlus || state.orbitMinus;

      // Reset the sweep accumulators on the rising edge of each toggle so the
      // motion begins with zero offset (i.e. exactly at the current camera).
      if (state.orbitActive && !prevOrbitActive) orbitYawAccum = 0.f;
      if (state.orbitPlus && !prevOrbitPlus)     orbitPlusPhase = 0.f;
      if (state.orbitMinus && !prevOrbitMinus)   orbitPlusPhase = 0.f;
      prevOrbitActive = state.orbitActive;
      prevOrbitPlus = state.orbitPlus;
      prevOrbitMinus = state.orbitMinus;

      {
        int loopMode = 0;
        if (uiServer->consumeRecordLoopRequest(loopMode)) {
          orbitPlusPhase = 0.f;
          loopTracking = true;
          loopTargetPhase = (loopMode == 1) ? 4.f * float(M_PI) : 2.f * float(M_PI);
          if (loopMode == 0) state.orbitMinus = true;
          else               state.orbitPlus = true;
        }
      }

      // The current interactive camera (WASD + mouse-look applied). Orbit and
      // plus modes use THIS as their starting placement, so toggling them on
      // sweeps around wherever you currently are instead of jumping back to the
      // initial/from-pose camera.
      const float sceneScale = glm::length(bb.diagonal());
      glm::mat4 R_pitch = glm::rotate(glm::radians(state.envRotationDegrees),  glm::vec3(1.f, 0.f, 0.f));
      glm::mat4 R_yaw   = glm::rotate(glm::radians(state.envRotationDegrees2), glm::vec3(0.f, 1.f, 0.f));
      glm::mat4 T_off   = glm::translate(glm::mat4(1.0f),
                                         -glm::vec3(state.X, state.Y, state.Z) * sceneScale);
      glm::mat4 interactiveView = R_pitch * R_yaw * T_off * viewMatrix;

      static const glm::mat4 kFlipZ = glm::mat4(  // 180 deg roll about view Z
          glm::vec4(-1.f, 0.f,  0.f, 0.f),
          glm::vec4( 0.f, -1.f, 0.f, 0.f),
          glm::vec4( 0.f, 0.f,  1.f, 0.f),
          glm::vec4( 0.f, 0.f,  0.f, 1.f));

      if (orbiting) {
        // We orbit around the point the camera is currently looking at (a point
        // ahead along the gaze, at the same distance as the scene centre) — NOT
        // the geometric centroid. Crucially we do NOT rebuild the view with
        // glm::lookAt (that picks its own up vector and can roll the camera
        // 180 deg). Instead we rigidly rotate the CURRENT view about the pivot
        // using the camera's own up/right axes, so at zero sweep the result is
        // exactly the interactive view — same place, same aim, same way up.
        const glm::mat4 camToWorld = glm::inverse(interactiveView);
        const glm::vec3 C = glm::vec3(camToWorld[3]);
        const glm::vec3 camRight = glm::normalize(glm::vec3(interactiveView[0][0],
                                                            interactiveView[1][0],
                                                            interactiveView[2][0]));
        const glm::vec3 camUp    = glm::normalize(glm::vec3(interactiveView[0][1],
                                                            interactiveView[1][1],
                                                            interactiveView[2][1]));
        const glm::vec3 fwd      = -glm::normalize(glm::vec3(interactiveView[0][2],
                                                            interactiveView[1][2],
                                                            interactiveView[2][2]));
        const float dist = glm::length(C) > 1e-4f ? glm::length(C) : sceneScale;
        const glm::vec3 pivot = C + dist * fwd;

        float yawOff = 0.f, pitchOff = 0.f;
        if (state.orbitMinus) {
          const float kSweepSpeedDeg = 1.5f * state.orbitSpeed;
          orbitPlusPhase += glm::radians(kSweepSpeedDeg);
          const float amp = glm::radians(state.orbitMinusAmpDeg);
          yawOff = amp * std::sin(orbitPlusPhase);
        } else if (state.orbitPlus) {
          const float kPlusSpeedDeg = 1.5f * state.orbitSpeed;
          orbitPlusPhase += glm::radians(kPlusSpeedDeg);
          const float twoPi = 2.0f * float(M_PI);
          const long  bar   = static_cast<long>(std::floor(orbitPlusPhase / twoPi));
          const float local = orbitPlusPhase - bar * twoPi;
          const float amp   = glm::radians(state.orbitPlusAmpDeg);
          if (bar % 2 == 0) pitchOff = amp * std::sin(local);  // vertical bar
          else              yawOff   = amp * std::sin(local);  // horizontal bar
        } else {
          float step = (orbitStepCLI != 0.f) ? orbitStepCLI : 1.0f;
          step *= state.orbitSpeed;
          orbitYawAccum += step;
          yawOff   = glm::radians(orbitYawAccum);
          pitchOff = glm::radians(state.orbitPitchDeg);
        }

        // Rigid rotation of the camera about the pivot, in the camera's frame:
        // yaw about its up axis (left/right), pitch about its right axis
        // (up/down). newView = oldView * T(pivot) * Q^-1 * T(-pivot).
        const glm::mat4 Qinv = glm::rotate(glm::mat4(1.f), -pitchOff, camRight)
                             * glm::rotate(glm::mat4(1.f), -yawOff, camUp);
        glm::mat4 orbitView = interactiveView
                            * glm::translate(glm::mat4(1.f), pivot)
                            * Qinv
                            * glm::translate(glm::mat4(1.f), -pivot);
        // Distance slider: slide the camera along its viewing axis. Positive
        // offset pulls the camera back from the pivot (view-space -Z).
        const float radiusDelta = state.orbitRadiusOffset * sceneScale;
        orbitView = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -radiusDelta)) * orbitView;
        if (state.flipCamera) orbitView = kFlipZ * orbitView;
        dynamicView = orbitView;

        if (loopTracking && orbitPlusPhase >= loopTargetPhase) {
          loopTracking = false;
          uiServer->sendLoopDone();
        }
      } else {
        dynamicView = interactiveView;
        if (state.flipCamera) dynamicView = kFlipZ * dynamicView;
      }

      // Convert OpenGL-style view (camera looks -Z, world +Y up on screen) to
      // COLMAP / 3DGS style (camera looks +Z, world +Y down in view) so the
      // same matrix + projection can be fed directly to diff-gaussian-
      // rasterization. This flips the Y and Z rows of dynamicView.
      static const glm::mat4 kOpenGLToColmap = glm::mat4(
          glm::vec4( 1.f,  0.f,  0.f, 0.f),
          glm::vec4( 0.f, -1.f,  0.f, 0.f),
          glm::vec4( 0.f,  0.f, -1.f, 0.f),
          glm::vec4( 0.f,  0.f,  0.f, 1.f));
      dynamicView = kOpenGLToColmap * dynamicView;

      
    } else {
      // Only log these if not in interactive mode:
      ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pts.size()/splatTimeSecs);
      ipu_utils::logger()->info("Splatted point count: {}", count);
    }

  } while (uiServer && state.stop == false);

  if (useDeviceLoop) {
    ipuSplatter->stopDeviceLoop();
  }

  hostProcessing.waitForCompletion();

  cv::imwrite("test.png", *imagePtr);

  return EXIT_SUCCESS;
}
