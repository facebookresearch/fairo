# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(torch_isolation)

add_subdirectory("pinocchio_isolation/")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find Torch
execute_process (
    COMMAND bash -c "python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())' | tr -d '\n'"
    OUTPUT_VARIABLE site_packages_path
)
set(Torch_DIR "${site_packages_path}/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

include_directories(
    "include"
    "${TORCH_INCLUDE_DIRS}"
    "${EIGEN3_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/pinocchio_isolation/include"
)

add_library(torch_server_ops SHARED src/torch_server_ops.cpp)
set_target_properties(torch_server_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torch_server_ops PRIVATE
    "${TORCH_LIBRARIES}"
)

add_library(torchrot SHARED src/rotations.cpp)
set_target_properties(torchrot PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torchrot PRIVATE
    "${TORCH_LIBRARIES}"
)

add_library(torchscript_pinocchio SHARED src/pinocchio.cpp)
set_target_properties(torchscript_pinocchio PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torchscript_pinocchio PRIVATE
    pinocchio_wrapper
    "${TORCH_LIBRARIES}"
    torchrot
)
