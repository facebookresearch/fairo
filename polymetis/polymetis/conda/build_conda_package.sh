#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
BASEDIR=$(dirname $0)

if [ -z "$PKG_PATH" ]; then 
    echo "Building Conda package, since PKG_PATH is not defined"
    # Rebuild package
    conda mambabuild -c pytorch -c fair-robotics -c aihabitat -c conda-forge $BASEDIR/conda_recipe
    echo "Input tar.bz2 file path shown above to automatically update the conda channel: "
    read PKG_PATH
fi

# Update channel directory
cp $PKG_PATH $BASEDIR/channel/linux-64
mamba index $BASEDIR/channel
