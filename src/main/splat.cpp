// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include "glm/matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>

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
  ("benchmark", po::value<int>()->default_value(0),
   "Run N frames headlessly with a fixed pose and report mean FPS, then exit. "
   "No --ui-port needed. Uses initial view or --from-pose if provided.");
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

  // --benchmark N: render N frames headlessly with a fixed pose and exit.
  const int benchmarkFrames = args["benchmark"].as<int>();
  if (benchmarkFrames > 0) {
    // Apply the same OpenGL→COLMAP flip used in the interactive loop:
    static const glm::mat4 kFlip = glm::mat4(
        glm::vec4( 1.f,  0.f,  0.f, 0.f),
        glm::vec4( 0.f, -1.f,  0.f, 0.f),
        glm::vec4( 0.f,  0.f, -1.f, 0.f),
        glm::vec4( 0.f,  0.f,  0.f, 1.f));
    auto benchView = kFlip * viewMatrix;

    ipuSplatter->updateModelView(benchView);
    ipuSplatter->updateProjection(projection);
    ipuSplatter->updateFocalLengths(state.fov, 0.f);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
      gm.execute(*ipuSplatter);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < benchmarkFrames; ++i) {
      gm.execute(*ipuSplatter);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double fps = benchmarkFrames / secs;

    printf("BENCHMARK: frames=%d total_sec=%.4f fps=%.2f ms_per_frame=%.3f\n",
           benchmarkFrames, secs, fps, 1000.0 * secs / benchmarkFrames);

    // Report per-phase cycle counts from the last frame
    {
      static constexpr int NP = 5;
      static const char* phaseNames[] = {
        "colourFb", "clearOutBuffers", "readInput_x4",
        "renderInternal", "total"
      };
      std::vector<unsigned> phTimes;
      ipuSplatter->getPhaseTimes(phTimes);
      // IPU Mk2 tile clock ~1.85 GHz (adjust if needed)
      double clockGHz = 1.85;
      printf("\nPHASE_TIMES (mean across %d tiles, from last frame):\n", fb.numTiles);
      for (int p = 0; p < NP; ++p) {
        double sum = 0;
        unsigned maxCycles = 0;
        for (int t = 0; t < fb.numTiles; ++t) {
          unsigned c = phTimes[t * NP + p];
          sum += c;
          if (c > maxCycles) maxCycles = c;
        }
        double meanCycles = sum / fb.numTiles;
        double meanMs = meanCycles / (clockGHz * 1e6);
        double maxMs  = maxCycles / (clockGHz * 1e6);
        printf("  %-20s mean_cycles=%10.0f  mean_ms=%7.4f  max_ms=%7.4f\n",
               phaseNames[p], meanCycles, meanMs, maxMs);
      }
    }

    ipuSplatter->getFrameBuffer(*imagePtr);
    cv::imwrite("benchmark_frame.png", *imagePtr);
    ipu_utils::logger()->info("Benchmark complete. Saved last frame to benchmark_frame.png");
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

  auto  dynamicView = viewMatrix;  
  do {
    auto startTime = std::chrono::steady_clock::now();
    *imagePtr = 0;
    std::uint32_t count = 0u;

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
      const float sceneScale = glm::length(bb.diagonal());
      glm::mat4 R_pitch = glm::rotate(glm::radians(state.envRotationDegrees),  glm::vec3(1.f, 0.f, 0.f));
      glm::mat4 R_yaw   = glm::rotate(glm::radians(state.envRotationDegrees2), glm::vec3(0.f, 1.f, 0.f));
      glm::mat4 T_off   = glm::translate(glm::mat4(1.0f),
                                         -glm::vec3(state.X, state.Y, state.Z) * sceneScale);
      dynamicView = R_pitch * R_yaw * T_off * viewMatrix;

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

  hostProcessing.waitForCompletion();

  cv::imwrite("test.png", *imagePtr);

  return EXIT_SUCCESS;
}
