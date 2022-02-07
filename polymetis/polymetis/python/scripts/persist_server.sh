#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

while true
do
        echo "Running $(which launch_robot.py)"
        $(which launch_robot.py) robot_client=franka_hardware &

        sleep 10
        while true
        do
                $(which ping_server.py)
                ret=$?
                if [ $ret -ne 0 ]; then
                    echo "=== Server died! Restarting server... ==="
                    echo "(Cleanup): Killing server and clients"
                    sudo pkill -9 run_server
                    sudo pkill -9 ".*franka.*"

                    break
                fi
                sleep 2
        done
done
