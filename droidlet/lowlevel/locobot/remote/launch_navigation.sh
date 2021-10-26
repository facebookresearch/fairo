#!/bin/env bash
set -ex


python slam_service.py &
timeout 10s bash -c "until python check_connected.py slam; do sleep 0.5; done;"

python planning_service.py &
timeout 10s bash -c "until python check_connected.py planner; do sleep 0.5; done;"

python navigation_service.py
timeout 10s bash -c "until python check_connected.py navigation; do sleep 0.5; done;"
