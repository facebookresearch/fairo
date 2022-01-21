#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail
trap "kill -- -$$" ERR # cleanup all child processes if error

export HYDRA_FULL_ERROR=1

######################
# Test empty statistics client
######################
# Start server & robot client
echo "========= Starting server and testing empty statistics client... ========="
launch_robot.py robot_client=empty_statistics_client use_real_time=false num_requests=5000 &
server_pid=$!
echo "========= Server PID: $server_pid ========="

# Wait
sleep 8

echo "========= Success. ========="

# Kill server
pkill -9 run_server || true
