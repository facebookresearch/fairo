#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail
trap "kill -- -$$" ERR # cleanup all child processes if error

######################
# Test empty statistics client
######################
# Start server & robot client
echo "=== Starting server and testing empty statistics client... ==="
launch_robot.py robot_client=empty_statistics_client use_real_time=false num_requests=5000 &
server_pid=$!
echo "=== Server PID: $server_pid ==="

# Wait
sleep 8

# Kill server
if ps -p $server_pid > /dev/null
then
    kill $server_pid
fi

echo "=== Success. ==="

######################
# Test simulation client
######################
# Start server & robot client
echo "=== Starting server and franka simulation...==="
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

######################
# Test mocked hardware client
######################
# Start server & robot client
echo "=== Starting server and mocked franka hardware client... ==="
launch_robot.py robot_client=franka_hardware use_real_time=false robot_client.executable_cfg.mock=true &
server_pid=$!
echo "=== Server PID: $server_pid ==="

sleep 4

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
    kill $server_pid
fi