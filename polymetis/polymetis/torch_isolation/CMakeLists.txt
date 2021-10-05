# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(polymetis)

SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Find Torch
execute_process (
    COMMAND bash -c "python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())' | tr -d '\n'"
    OUTPUT_VARIABLE site_packages_path
)
set(Torch_DIR ${site_packages_path}/torch/share/cmake/Torch/)
find_package(Torch REQUIRED)

include_directories(
    include
    "${CMAKE_CURRENT_BINARY_DIR}"
    "$ENV{CONDA_PREFIX}/include/"
    ${TORCH_INCLUDE_DIRS}
)

