#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -x -e

# Default values
if [ -z "$CFG" ]; then CFG="Release"; fi
if [ -z "$DEV_PYTHON" ]; then DEV_PYTHON=ON; fi
if [ -z "$PREFIX" ]; then PREFIX=$CONDA_PREFIX; fi
if [ -z "$PYTHON" ]; then PYTHON=python; fi
if [ -z "$BUILD_TESTS" ]; then BUILD_TESTS=ON; fi
if [ -z "$BUILD_DOCS" ]; then BUILD_DOCS=OFF; fi
if [ -z "$BUILD_FRANKA" ]; then BUILD_FRANKA=OFF; fi
if [ -z "$BUILD_ALLEGRO" ]; then BUILD_ALLEGRO=OFF; fi
if [ -z "$REBUILD_LIBFRANKA" ]; then REBUILD_LIBFRANKA=OFF; fi

# Build libfranka
# (Note: Build if BUILD_FRANKA=ON but libfranka is not built locally, or if forced by setting REBUILD_LIBFRANKA)
LIBFRANKA_PATH="src/clients/franka_panda_client/third_party/libfranka"
BUILD_PATH="${LIBFRANKA_PATH}/build"

if [ ! -d "$BUILD_PATH" ]; then 
    if [ "$BUILD_FRANKA" == "ON" ]; then 
        REBUILD_LIBFRANKA="ON";
    fi
fi

if [ "$REBUILD_LIBFRANKA" == "ON" ]; then
    if [ -d "$BUILD_PATH" ]; then rm -r $BUILD_PATH; fi
    mkdir -p $BUILD_PATH && cd $BUILD_PATH
    cmake -DCMAKE_BUILD_TYPE=$CFG -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$PREFIX/lib \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PREFIX/lib \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$PREFIX/bin ..
    cmake --build .
    cd -
fi

# Build Franka clients if libfranka is built
if [ -d "$BUILD_PATH" ]; then 
    BUILD_FRANKA="ON";
fi

# Build c++ 
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$CFG -DBUILD_FRANKA=$BUILD_FRANKA -DBUILD_TESTS=$BUILD_TESTS -DBUILD_DOCS=$BUILD_DOCS \
    -DBUILD_ALLEGRO=$BUILD_ALLEGRO \
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$PREFIX/lib \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PREFIX/lib \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$PREFIX/bin ..
cmake --build .

cd ..

# Install python package
if [ "$DEV_PYTHON" == "ON" ]; then
    $PYTHON -m pip install -vvv -e .
else
    $PYTHON -m pip install -vvv .
fi
