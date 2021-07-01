#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail
trap "kill -- -$$" ERR # cleanup all child processes if error

echo "=== Starting server and testing empty statistics client... ==="
launch_robot.py robot_client=empty_statistics_client use_real_time=false num_requests=5000 &
server_pid=$!
echo "=== Server PID: $server_pid ==="

sleep 8

if ps -p $server_pid > /dev/null
then
    kill $server_pid
fi

echo "=== Success. ==="

echo "=== Starting server and franka simulation...==="
# Start server & robot client
launch_robot.py robot_client=franka_sim gui=false use_real_time=false &
server_pid=$!
echo "=== Server PID: $server_pid ==="

sleep 4

# Run RobotInterface tests (previous agent.py tests)
echo "=== Running controller tests in simulation to check tracking... ==="
for entry in "tests/scripts"/*; do
    if [ ${entry: -3} == ".py" ]; then
        echo $entry
        python $entry
    fi
done
echo "=== Success. ==="

# Run examples
echo "=== Running controllers in examples/ ... ==="
for entry in "examples"/*; do
    if [ ${entry: -3} == ".py" ]; then
        echo $entry
        python $entry
    fi
done
echo "=== Success. ==="

# Kill server
if ps -p $server_pid > /dev/null
then
    echo "Killing server_pid $server_pid"
    kill $server_pid
fi
