#!/usr/bin/env bash
set -ex

python slam_service.py $CAMERA_NAME &> slam_service.out &
timeout 20s bash -c "until python check_connected.py slam $PYRO_IP; do sleep 0.5; done;" || true

python planning_service.py &> planning_service.out &
timeout 20s bash -c "until python check_connected.py planner $PYRO_IP; do sleep 0.5; done;" || true

python navigation_service.py $ROBOT_NAME &> navigation_service.out &
timeout 20s bash -c "until python check_connected.py navigation $PYRO_IP; do sleep 0.5; done;" || true
