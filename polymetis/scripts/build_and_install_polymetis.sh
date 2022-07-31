#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

GIT_ROOT=$(git rev-parse --show-toplevel)
POLYMETIS_PATH="$GIT_ROOT/polymetis/polymetis"

# Install polymetis
cd $POLYMETIS_PATH
BUILD_FRANKA=ON
source install.sh
cd -