#!/bin/env bash
set -e

if $(which python | grep fairo >/dev/null); then
    echo "droidlet virtualenv needs to be deactivated before running this script. Run 'deactivate' in the terminal before running this"
    exit 0
fi
roslaunch stretch_laser_odom_base.launch &
BGPID=$!
trap 'echo "Killing $BGPID"; kill $BGPID; exit' INT
sleep 5
roslaunch stretch_hector_slam.launch
