#!/usr/bin/env bash
set -ex

python -u slam_service.py $CAMERA_NAME &> slam_service.out &
timeout --foreground 20s bash -c "until python check_connected.py slam $PYRO_IP; do sleep 0.5; done;" || true

python -u planning_service.py &> planning_service.out &
timeout --foreground 20s bash -c "until python check_connected.py planner $PYRO_IP; do sleep 0.5; done;" || true

python -u navigation_service.py $ROBOT_NAME &> navigation_service.out &
timeout --foreground 20s bash -c "until python check_connected.py navigation $PYRO_IP; do sleep 0.5; done;" || true
