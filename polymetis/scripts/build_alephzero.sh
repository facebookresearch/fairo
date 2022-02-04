#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

GIT_ROOT=$(git rev-parse --show-toplevel)
ALEPHZERO_PATH="$GIT_ROOT/polymetis/polymetis/third_party/alephzero"

# Ensure submodules exist
git submodule update --init --recursive

# Build
cd $ALEPHZERO_PATH
mkdir -p build
PREFIX=./build make install -j
