#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail
trap "kill -- -$$" ERR # cleanup all child processes if error

export HYDRA_FULL_ERROR=1

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)

######################
# Test mocked hardware client
######################
# Start server & robot client
echo "========= Starting server and mocked franka hardware client... ========="
launch_robot.py robot_client=franka_hardware use_real_time=false robot_client.executable_cfg.mock=true &
server_pid=$!
echo "========= Server PID: $server_pid ========="

sleep 4

# Run examples
echo "========= Running controllers in examples/ ... ========="
for entry in "$PROJECT_ROOT_DIR/polymetis/examples"/*; do
    if [ ${entry: -3} == ".py" ]; then
        echo "====== Running 'python $entry' ======"
        python $entry
    fi
done

echo "========= Success. ========="

# Kill server
pkill -9 run_server || true
