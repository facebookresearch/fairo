#!/bin/env bash
set -ex

python slam_service.py &
timeout 10s bash -c "until python check_connected.py slam $LOCOBOT_IP; do sleep 0.5; done;" || true

python planning_service.py &
timeout 10s bash -c "until python check_connected.py planner $LOCOBOT_IP; do sleep 0.5; done;" || true

python navigation_service.py &
timeout 10s bash -c "until python check_connected.py navigation $LOCOBOT_IP; do sleep 0.5; done;" || true
