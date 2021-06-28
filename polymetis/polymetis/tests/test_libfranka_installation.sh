#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

FILE="$CONDA_PREFIX/lib/libfranka.so"

echo "Testing for ${FILE}..."
if [ -f $FILE ]; then
    echo "Found"
    exit 0
else
    echo "Error: Cannot locate libfranka.so"
    exit 1
fi