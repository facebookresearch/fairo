#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

GIT_ROOT=$(git rev-parse --show-toplevel)

LIBFRANKA_PATH="$GIT_ROOT/polymetis/src/clients/franka_panda_client/third_party/libfranka"

# Check to make sure directory exists
[ ! -d $LIBFRANKA_PATH ] && echo "Directory $LIBFRANKA_PATH does not exist" && exit 1

# Ensure submodules exist
git submodule update --init --recursive

# Build
BUILD_PATH="${LIBFRANKA_PATH}/build"
mkdir -p $BUILD_PATH && cd $BUILD_PATH
echo "Building libfranka at $BUILD_PATH"

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
cmake --build .

cd -
