
message(STATUS "Adding splat library: ${CMAKE_CURRENT_SOURCE_DIR}")

FILE(GLOB_RECURSE LIB_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/splat/*.hpp)

FILE(GLOB_RECURSE LIB_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tileMapping/*.hpp)
FILE(GLOB_RECURSE LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(splat_utils ${LIB_INC} ${LIB_SRC})

# OpenMP parallelises the Stage-2 gather discovery (computeGatherOffsets over
# all Gaussians each frame). PUBLIC so the -fopenmp flag reaches the header in
# consumers too.
target_link_libraries(splat_utils PUBLIC OpenMP::OpenMP_CXX)
