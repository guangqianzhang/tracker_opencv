cmake_minimum_required(VERSION 3.17)
project(cvnn3)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-std=c++14)
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -fopenmp")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fopenmp")
find_package(OpenCV 4.5.5 REQUIRED)
message("opencv_dir=" ${OpenCV_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
include_directories(${YAML_CPP_INCLUDEDIR})

include_directories(include
        ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
message("OpenCV_CONFIG_PATH  is:" ${OpenCV_CONFIG_PATH})
message("OpenCV_INSTALL_PATH is:" ${OpenCV_INSTALL_PATH})

add_executable(cvnn3 main.cpp Yolo_plug.hpp detect_img_keypoints.cpp match.cpp  )
target_link_libraries (cvnn3 ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES} )