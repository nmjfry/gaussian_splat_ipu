cmake_minimum_required(VERSION 3.16)
project(gaussian_splat_ipu)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(utils.cmake)
check_for_submodules()
get_poplar_version()
message(STATUS "Detected Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
set(GLM_INCLUDE_DIRS external/glm)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GLM_INCLUDE_DIRS}
  ${IPU_INCLUDES}
  ${PROJECT_SOURCE_DIR}/include
)

set(POPLAR_LINK_LIBRARIES -lpoplin -lpopnn -lpoplar -lpopops -lpoputil -lpoprand -lgcl -lpvti)

# popc won't accept relative path from within Poplar so we define this absolute
# path to give us a prefix to start from:
add_compile_definitions(POPC_PREFIX="${PROJECT_SOURCE_DIR}")

# External Libraries:
add_subdirectory(external/packetcomms)
add_subdirectory(external/videolib)

include_directories(
  ${PACKETCOMMS_INCLUDES}
  ${VIDEOLIB_INCLUDES}
)

# Internal libraries:
add_subdirectory(${PROJECT_SOURCE_DIR}/src/splat)

# Executables:
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/main)
