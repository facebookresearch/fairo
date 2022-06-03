# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
CFG="Release"
DEV_PYTHON="OFF"
BUILD_FRANKA="ON"
BUILD_ALLEGRO="ON"
REBUILD_LIBFRANKA="ON"

cd ./polymetis/polymetis
rm -rf build
source install.sh