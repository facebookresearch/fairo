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

include_directories(
    "include"
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(torch_server_ops SHARED src/torch_server_ops.cpp)
target_link_libraries(torch_server_ops
    ${TORCH_LIBRARIES}
)

add_library(torchrot SHARED src/rotations.cpp)
target_link_libraries(torchrot
    ${TORCH_LIBRARIES}
)

add_library(torchscript_pinocchio SHARED src/pinocchio.cpp)
target_link_libraries(torchscript_pinocchio
    /private/home/yixinlin/dev/fairo/polymetis/polymetis/pinocchio_isolation/build/libpinocchio_wrapper.so
    ${TORCH_LIBRARIES}
    torchrot
)
