# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(torch_isolation)

SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find Torch
set(Torch_DIR "/private/home/yixinlin/Downloads/libtorch-1.10-old-abi/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Pinocchio
set(pinocchio_DIR $ENV{CONDA_PREFIX}/lib/cmake/pinocchio)
find_package(pinocchio REQUIRED)

include_directories(
    "include"
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${pinocchio_INCLUDE_DIRS} 
)

add_library(torch_server_ops src/torch_server_ops.cpp)
set_target_properties(torch_server_ops PROPERTIES CXX_VISIBILITY_PRESET default)

add_library(torchrot src/rotations.cpp)
set_target_properties(torchrot PROPERTIES CXX_VISIBILITY_PRESET default)

add_library(torchscript_pinocchio src/pinocchio.cpp)
target_link_libraries(torchscript_pinocchio
    ${pinocchio_LIBRARIES}
    torchrot
)
# set_target_properties(torch_server_ops PROPERTIES CXX_VISIBILITY_PRESET default)
