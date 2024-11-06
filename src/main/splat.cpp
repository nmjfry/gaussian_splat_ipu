// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include <opencv2/highgui.hpp>

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
   "Disable use of optimised AMP codelets.");
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

  // Translate all points so the centroid is zero then negate the z-axis:
  {
    const auto bbCentre = bb.centroid();
    for (auto& v : pts) {
      v.p -= bbCentre;
      v.p.z = -v.p.z;
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
    g.mean = {pt.x, pt.y, pt.z, 1.f};
    if (ply.f_dc[0].values.size() > 0) {
      glm::vec3 colour = {SH_C0 * ply.f_dc[0].values[i],
                      SH_C0 * ply.f_dc[1].values[i],
                      SH_C0 * ply.f_dc[2].values[i]};
      colour += 0.5f;
      colour = glm::max(colour, glm::vec3(0.f));
      g.colour = {colour.x, colour.y, colour.z, ply.opacity.values[i]};
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

  auto FOV = glm::radians(40.f);

  // Setup a user interface server if requested:
  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  state.fov = glm::radians(40.f);
  state.device = args.at("device").as<std::string>();
  auto uiPort = args.at("ui-port").as<int>();
  if (uiPort) {
    uiServer.reset(new InterfaceServer(uiPort));
    uiServer->start();
    uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    uiServer->updateFov(state.fov);
  }

  // Set up the modelling and projection transforms in an OpenGL compatible way:
  auto modelView = splat::lookAtBoundingBox(bb, glm::vec3(0.f , 1.f, 1.f), 1.f);

  // Transform the BB to camera/eye space:
  splat::Bounds3f bbInCamera(
    modelView * glm::vec4(bb.min, 1.f),
    modelView * glm::vec4(bb.max, 1.f)
  );

  ipu_utils::logger()->info("Point bounds (eye space): {}", bbInCamera);
  auto projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);

  ipuSplatter->updateModelView(modelView);
  ipuSplatter->updateProjection(projection);
  gm.prepareEngine();

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

  auto  dynamicView = modelView;  
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
        // print viewmatrix
        for (int i = 0; i < 4; i++) {
          ipu_utils::logger()->info("Dynamic view matrix: {} {} {} {}", modelView[i][0], modelView[i][1], modelView[i][2], modelView[i][3]);
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

      dynamicView = glm::translate(modelView, glm::vec3(0.f, 0.f , state.Z));
      dynamicView = glm::translate(dynamicView, glm::vec3(state.X,  state.Y , 0.f));
      dynamicView = glm::rotate(dynamicView, glm::radians(state.envRotationDegrees), glm::vec3(1.f, 0.f, 0.f));
      dynamicView = glm::rotate(dynamicView, glm::radians(state.envRotationDegrees2), glm::vec3(0.f, 1.f, 0.f));


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
