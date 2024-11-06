// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/ipu_rasteriser.hpp>
#include <splat/geometry.hpp>
#include <ipu/io_utils.hpp>
#include <opencv2/highgui.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <poputil/TileMapping.hpp>

#include <popops/codelets.hpp>
#include <popops/TopK.hpp>
#include <popops/Fill.hpp>

#include <tileMapping/edge_builder.hpp>

using namespace poplar;

namespace splat {

IpuSplatter::IpuSplatter(const Points& verts, TiledFramebuffer& fb, bool noAMP)
  : modelView("mv"), projection("mp"), fxy("fxy"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"), 
    counts("splat_counts"),
    hostModelView(16),
    hostProjection(16),
    fxyHost(2),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  hostVertices.reserve(4 * verts.size());
  printf("VERTS size: %luB\n", verts.size());
  for (const auto& v : verts) {
    hostVertices.push_back(v.p.x);
    hostVertices.push_back(v.p.y);
    hostVertices.push_back(v.p.z);
    hostVertices.push_back(1.f);
  }
  frameBuffer.reserve(fb.width * fb.height * 4);
  for (uint i = 0; i < fb.width * fb.height; ++i) {
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
  }
  printf("Fb size: %luB\n", frameBuffer.size());
}

IpuSplatter::IpuSplatter(const Gaussians& verts, TiledFramebuffer& fb, bool noAMP)
  : modelView("mv"), projection("mp"), fxy("fxy"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"), 
    counts("splat_counts"),
    hostModelView(16),
    hostProjection(16),
    fxyHost(2),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  auto elemSize = sizeof(verts[0]);
  hostVertices.reserve(elemSize * verts.size());
  printf("num verts in: %lu, elemsize: %lu \n", verts.size(), elemSize);
  
  for (auto j = 0u; j < verts.size(); ++j) {
    auto gptr = (const float*)&verts[j];
    for (auto i = 0u; i < elemSize; ++i) {
      hostVertices.push_back(*(gptr + i));
    }
  }

  frameBuffer.reserve(fb.width * fb.height * 4);
  for (uint i = 0; i < fb.width * fb.height; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      frameBuffer.push_back(0.0);
    }
  }

  printf("Fb size: %luB\n", frameBuffer.size());

  splatCounts.resize(fb.numTiles);
  for (auto& c : splatCounts) {
    c = 0;
  }
}


void IpuSplatter::updateModelView(const glm::mat4& mv) {
  auto mvt = glm::transpose(mv);
  auto ptr = (const float*)glm::value_ptr(mvt);
  for (auto i = 0u; i < hostModelView.size(); ++i) {
    hostModelView[i] = *ptr;
    ptr += 1;
  }
}

void IpuSplatter::updateProjection(const glm::mat4& mp) {
  auto mpt = glm::transpose(mp);
  auto ptr = (const float*)glm::value_ptr(mpt);
  for (auto i = 0u; i < hostProjection.size(); ++i) {
    hostProjection[i] = *ptr;
    ptr += 1;
  }
}

void IpuSplatter::getIPUHistogram(std::vector<u_int32_t>& counts) const {
  counts = splatCounts;
}

void IpuSplatter::updateFocalLengths(float fx, float fy) {
  fxyHost = {fx, fy};
}

// takes a cv::Mat image and returns a copy of the original but with the image partitioned into tiles of size tileHeight x tileWidth.
// dataType is the data type of the image (e.g. CV_8UC3 for 8-bit unsigned char 3-channel image)
// It treats the image as one vector of pixels which we pick from in chunks of tileHeight x tileWidth
cv::Mat tileImageBuffer(cv::Mat image, int tileHeight, int tileWidth, int dataType, int channels) {
    cv::Mat new_image(image.rows, image.cols, dataType);
    uchar *buffer = image.data;
    int stripSize = tileHeight * tileWidth;
     
    for (int j = 0; j < int(floor(image.rows / float(tileHeight))); j++) {
        for (int i = 0; i < int(floor(image.cols / float(tileWidth))); i++) {
            cv::Mat chunk(tileHeight, tileWidth, dataType, buffer, cv::Mat::AUTO_STEP);
            chunk.copyTo(new_image(cv::Rect(i * tileWidth, j * tileHeight, tileWidth, tileHeight)));
            buffer += stripSize * channels;
        }
    }

    return new_image;
}

void IpuSplatter::getFrameBuffer(cv::Mat &frame) const {
  // need to ensure that we read sections of the framebuffer
  // as square tiles and then stitch them together

  cv::Mat image_f = cv::Mat(cv::Size(fbMapping.width, fbMapping.height), CV_32FC4, (void *) frameBuffer.data(), cv::Mat::AUTO_STEP);
  // Clamp values between 0 and 255

  cv::Mat image_f_8u;
  cv::min(image_f * 255.0f, 255.0f, image_f);
  image_f.convertTo(image_f_8u, CV_8UC4);
  cvtColor(image_f_8u, frame, cv::COLOR_RGBA2BGR);
  frame = tileImageBuffer(frame, fbMapping.tileHeight, fbMapping.tileWidth, CV_8UC3, 3);

}

void IpuSplatter::getProjectedPoints(std::vector<glm::vec4>& pts) const {
  pts.resize(hostVertices.size() / 4);
  const auto* ptr = hostVertices.data();
  for (auto i = 0u; i < pts.size(); ++i) {
    pts[i].x = *(ptr + 0);
    pts[i].y = *(ptr + 1);
    pts[i].z = *(ptr + 2);
    pts[i].w = *(ptr + 3);
    ptr += 4;
  }
}

struct MappingInfo {
  std::size_t padding;
  std::size_t elementsPerTile;
  std::size_t totalTiles;
};

MappingInfo calculateMapping(poplar::Graph& g, std::size_t numElements, std::size_t grainSize, TiledFramebuffer &fbMapping) {
  ipu_utils::logger()->info("Input size of data: {}B", numElements);
  const double numTiles = g.getTarget().getNumTiles();

  if (fbMapping.numTiles < numTiles) {
    ipu_utils::logger()->info("Number of tiles in framebuffer ({}) is less than number of tiles on target ({})", fbMapping.numTiles, numTiles);
  }

  double grainsPerTile = std::ceil(numElements / (fbMapping.numTiles * grainSize));
  double elementsPerTile = grainsPerTile * grainSize;
  double fullTiles = std::floor(numElements / elementsPerTile);
  double unfilledTiles = fbMapping.numTiles - fullTiles;
  double remainingElements = numElements - (fullTiles * elementsPerTile);
  double paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;

  auto totalTiles = fullTiles + unfilledTiles;
  totalTiles = totalTiles > fbMapping.numTiles - 1 ? fbMapping.numTiles - 1 : totalTiles;

  ipu_utils::logger()->info("Upper bound elements per tile: {}", elementsPerTile);
  ipu_utils::logger()->info("Full tiles: {}", fullTiles);
  ipu_utils::logger()->info("Unfilled tiles: {}", unfilledTiles);
  ipu_utils::logger()->info("Remaining elements: {}", remainingElements);
  ipu_utils::logger()->info("Padded elements on last used tile: {}", paddedRemainder);
  ipu_utils::logger()->info("Padding: {}", paddedRemainder - remainingElements);
  ipu_utils::logger()->info("Total padding to fill all tiles: {}", paddedRemainder - remainingElements + (unfilledTiles * elementsPerTile));
  ipu_utils::logger()->info("Total tiles: {}", totalTiles);

  const std::size_t padding = paddedRemainder - remainingElements + (unfilledTiles * elementsPerTile);
  return MappingInfo{padding, std::size_t(elementsPerTile), std::size_t(totalTiles)};
}


/// Distribute elements across tiles such that the number of elements on a
/// tile is always divisible by grainSize. The tensor must have already been padded
/// to a multiple of grain size.
void applyTileMapping(poplar::Graph& g, const poplar::Tensor& paddedInput, const MappingInfo& info) {
  auto sliceStart = 0u;
  auto t = 0u;
  for (t; t < info.totalTiles; ++t) {
    const auto sliceEnd = sliceStart + info.elementsPerTile;
    g.setTileMapping(paddedInput.slice(sliceStart, sliceEnd), t);
    sliceStart = sliceEnd;
  }

  // Last tile has fewer elements:
  auto lastSlice = paddedInput.slice(sliceStart, paddedInput.numElements());
  ipu_utils::logger()->info("Size of slice on last tile: {}", lastSlice.numElements());
  if (lastSlice.numElements() > 0) {
    g.setTileMapping(lastSlice, t);
  }
}

// Add a vertex to project vertices that uses vanilla C++ code.
void addProjectionVertex(poplar::Graph& g, poplar::ComputeSet& cs, unsigned t, const poplar::Tensor& tid,  const poplar::Tensor& westIn, const poplar::Tensor& eastOut,
                         const poplar::Tensor& modelViewProjection, const poplar::Tensor& ptsIn, const poplar::Tensor& localFb) {
  auto v = g.addVertex(cs, "GSplat");
  g.setTileMapping(v, t);

  g.connect(v["matrix"], modelViewProjection);
  g.connect(v["vertsIn"], ptsIn);
  g.connect(v["localFb"], localFb);
  g.connect(v["tile_id"], tid);
  g.connect(v["westIn"], westIn);
  g.connect(v["eastOut"], eastOut);

}

// Add a vertex to project vertices that uses is optimised using the tile's AMP engine.
void addProjectionVertexAMP(poplar::Graph& g, poplar::ComputeSet& cs, unsigned t,
                        const poplar::Tensor& modelViewProjection, const poplar::Tensor& sliceIn, const poplar::Tensor& sliceOut) {
  auto v = g.addVertex(cs, "Transform4x4_amp");
  auto vs = g.addVertex(cs, "LoadMatrix");
  g.setTileMapping(v, t);
  g.setTileMapping(vs, t);

  g.connect(vs["matrix"], modelViewProjection);
  g.connect(v["vertsIn"], sliceIn);
  g.connect(v["vertsOut"], sliceOut);

  const auto vertsThisTile = sliceIn.numElements() / 4;
  if (vertsThisTile % 8 != 0) {
    ipu_utils::logger()->error("Tile {} has {} vertices which is not a multiple of 8", t, vertsThisTile);
    throw std::runtime_error("Vertices per tile must be a multiple of 8 to use the AMP.");
  }
}

void IpuSplatter::build(poplar::Graph& graph, const poplar::Target& target) {
  auto vg = graph.createVirtualGraph(0u, fbMapping.numTiles);

  const auto codeletFile = std::string(POPC_PREFIX) + "/codelets/splat/codelets.cpp";
  const auto glmPath = std::string(POPC_PREFIX) + "/external/glm/";
  const auto mathPath = std::string(POPC_PREFIX) + "/include/math";
  const auto otherIncludes = std::string(POPC_PREFIX) + "/include/missing";
  const auto tileMapping = std::string(POPC_PREFIX) + "/include/tileMapping";
  const auto includes = " -I " + glmPath + " -I " + mathPath + " -I " + otherIncludes + " -I " + tileMapping;
  ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
  popops::addCodelets(vg);
  vg.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);

  // Create storage for the model view projeciton matrix. Place the master copy on tile 0
  // and then broadcast from their to all other tiles before any computations.
  modelView.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(modelView, 0u);

  projection.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(projection, 0u);

  fxy.buildTensor(vg, FLOAT, {2});
  vg.setTileMapping(fxy, 0u);

  // Build a program to upload and broadcast the modelling-projection matrix:
  program::Sequence broadcastMvp;
  broadcastMvp.add(modelView.buildWrite(vg, true));
  broadcastMvp.add(projection.buildWrite(vg, true));
  broadcastMvp.add(fxy.buildWrite(vg, true));

  auto fbGrainSize = 4;
  auto fbToTileMapping = calculateMapping(vg, frameBuffer.size(), fbGrainSize, fbMapping);
  ipu_utils::logger()->info("Framebuffer layout: padding: {}, elementsPerTile: {}, totalTiles: {}", fbToTileMapping.padding, fbToTileMapping.elementsPerTile, fbToTileMapping.totalTiles);
  auto paddedFramebuffer = vg.addVariable(FLOAT, {frameBuffer.size() + fbToTileMapping.padding}, "padded_frame_buffer");
  applyTileMapping(vg, paddedFramebuffer, fbToTileMapping);
  outputFramebuffer = paddedFramebuffer.slice(0, frameBuffer.size());

  // Map the point cloud vertices across all tiles. TODO: If we are not using AMP the only constraint
  // is that the grain size must be a multiple of 4 (so that 4-vectors are not split between
  // tiles). If we use the AMP we need to have at least 8 4-vectors to fill the AMP pipeline so
  // the minimum grain size is 32:
  ipu_utils::logger()->info("hostvertices size: {}, numtiles {}", hostVertices.size(), fbToTileMapping.totalTiles);
  const auto grainSize = sizeof(Gaussian3D); //disableAMPVertices ? 4 : 4 * 8;
  auto mapping = calculateMapping(vg, hostVertices.size(), grainSize, fbMapping);
  ipu_utils::logger()->info("Vertex layout: padding: {}, elementsPerTile: {}, totalTiles: {}", mapping.padding, mapping.elementsPerTile, mapping.totalTiles);
  auto paddedInput = vg.addVariable(FLOAT, {hostVertices.size() + mapping.padding}, "padded_verts_in");
  applyTileMapping(vg, paddedInput, mapping);
  // We only want to stream to a slice of the padded tensor:
  inputVertices = paddedInput.slice(0, hostVertices.size());

  // Build a compute set to transform the points:
  const auto csName = disableAMPVertices ? "project" : "project_amp";
  auto splatCs = vg.addComputeSet(csName);

// for TUM desk:
// 450
// chan * 2
  unsigned numPoints = 350;
  std::size_t channelSize = numPoints * grainSize;
  std::size_t extraStorageSize = channelSize * 2;

  // construct z-buffer program to sort the gaussians
  program::Sequence sortGaussians;

  MappingInfo zBufferMapping = mapping;
  zBufferMapping.elementsPerTile = (zBufferMapping.elementsPerTile + extraStorageSize) / grainSize;
  std::size_t totalGaussianCapacity = (hostVertices.size() + mapping.padding + extraStorageSize * zBufferMapping.totalTiles) / grainSize;
  
  const auto indices = vg.addVariable(poplar::INT, {totalGaussianCapacity}, "indices");
  applyTileMapping(vg, indices, zBufferMapping);

  const auto splatCounts = vg.addVariable(poplar::UNSIGNED_INT, {(size_t) fbMapping.numTiles});
  MappingInfo counterInfo = {0, 1, (size_t) fbMapping.numTiles};
  applyTileMapping(vg, splatCounts, counterInfo);
  counts = splatCounts.slice(0, fbMapping.numTiles);


  std::vector<poplar::VertexRef> vertices;
  // Get the tile mapping and connect the vertices:
  const auto tm = vg.getTileMapping(paddedInput);
  const auto tmFb = vg.getTileMapping(paddedFramebuffer);
  const auto tmIndices = vg.getTileMapping(indices);
  const auto tmCounts = vg.getTileMapping(splatCounts);

  for (auto t = 0u; t < tm.size(); ++t) {
    const auto& m = tm[t];
    const auto& mFb = tmFb[t];
    const auto& mIndices = tmIndices[t];
    const auto& mCounts = tmCounts[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected fb to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      // Add the tile local MVP matrix variable and append a copies that broadcast it to all tiles:
      auto localMv = vg.clone(modelView, "mv_tile_" + std::to_string(t));
      vg.setTileMapping(localMv, t);
      broadcastMvp.add(program::Copy(modelView, localMv));

      auto localProj = vg.clone(projection, "mp_tile_" + std::to_string(t));
      vg.setTileMapping(localProj, t);
      broadcastMvp.add(program::Copy(projection, localProj));

      auto localFxy = vg.clone(fxy, "fxy_tile_" + std::to_string(t));
      vg.setTileMapping(localFxy, t);
      broadcastMvp.add(program::Copy(fxy, localFxy));

      auto ptsIn = paddedInput.slice(m.front());
      auto sliceFb = paddedFramebuffer.slice(mFb.front());
      auto sliceIdxs = indices.slice(mIndices.front());
      auto counter = splatCounts.slice(mCounts.front());

      auto storage = vg.addVariable(poplar::FLOAT, {extraStorageSize}, "extra_storage");
      vg.setTileMapping(storage, t);
      auto gaussians = concat(ptsIn, storage);

      auto gaus2D = vg.addVariable(poplar::FLOAT, {sliceIdxs.numElements() * sizeof(Gaussian2D)}, "z_buffer");
      vg.setTileMapping(gaus2D, t);

      auto tid = vg.addConstant<int>(INT, {1}, {int(t)});
      vg.setTileMapping(tid, t);


      auto v = vg.addVertex(splatCs, "GSplat");
      vg.setTileMapping(v, t);
      vg.connect(v["modelView"], localMv.flatten());
      vg.connect(v["projection"], localProj.flatten());
      vg.connect(v["vertsIn"], gaussians);
      vg.connect(v["indices"], sliceIdxs);
      vg.connect(v["gaus2D"], gaus2D);
      vg.connect(v["localFb"], sliceFb);
      vg.connect(v["fxy"], localFxy);
      vg.connect(v["tile_id"], tid);
      vg.connect(v["splatted"], counter);  
      vertices.push_back(v);
    }
  }


  EdgeBuilder eb(vg, vertices, channelSize);
  eb.constructLattice(tm, fbMapping);
  // this program sequence will copy the points between all the tiles in the graph
  program::Sequence broadcastPoints = eb.getBroadcastSequence();

  program::Sequence main;
  main.add(broadcastMvp); // sends the model view and projection matrices to all tiles
  main.add(program::Execute(splatCs)); // splats the gaussians
  main.add(broadcastPoints); // broadcasts any misplaced gaussians to other tiles

  main.add(outputFramebuffer.buildRead(vg, true));
  main.add(counts.buildRead(vg, true));

  program::Sequence setup;
  setup.add(inputVertices.buildWrite(vg, true));

  getPrograms().add("write_verts", setup);
  getPrograms().add("project", main);
}

void IpuSplatter::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!initialised) {
    initialised = true;
    modelView.connectWriteStream(engine, hostModelView);
    projection.connectWriteStream(engine, hostProjection);
    fxy.connectWriteStream(engine, fxyHost);
    inputVertices.connectWriteStream(engine, hostVertices);
    outputFramebuffer.connectReadStream(engine, frameBuffer);
    counts.connectReadStream(engine, splatCounts);
    getPrograms().run(engine, "write_verts");
  }
  getPrograms().run(engine, "project");
}

} // end of namespace splat
