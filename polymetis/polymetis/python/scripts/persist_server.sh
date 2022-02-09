#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT TERM

function cleanup() {
  echo "=== (Cleanup): Killing server and clients ==="
  pkill -9 "run_server"
  pkill -9 "franka_panda_cl"
}

function ctrl_c() {
  cleanup
  exit 130
}

. /home/box/miniconda3/etc/profile.d/conda.sh
conda activate polymetis-local
$(which ping_server.py) > ping_server.out &
ret=$?
if [ $ret -ne 0 ]; then
  echo "Unable to start ping_server.py"
  exit 1
fi


while true
do
  echo "=== Running $(which launch_robot.py) ==="
  $(which launch_robot.py) robot_client=franka_hardware &

  sleep 10
  while true
  do
    $(which last_state_timestep.py)
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "=== Server died! Restarting server... ==="
        cleanup
        break
    fi
    sleep 2
  done
done
