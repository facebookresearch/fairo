#!/usr/bin/env bash
set -ex

python slam_service.py $CAMERA_NAME &
timeout 10s bash -c "until python check_connected.py slam $PYRO_IP; do sleep 0.5; done;" || true

python planning_service.py &
timeout 10s bash -c "until python check_connected.py planner $PYRO_IP; do sleep 0.5; done;" || true

python navigation_service.py $ROBOT_NAME &
timeout 10s bash -c "until python check_connected.py navigation $PYRO_IP; do sleep 0.5; done;" || true
