#!/usr/bin/env bash
set -ex


DROIDLET_DIR="$dirname $(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))"
SEMANTIC_PREDICTION_RELDIR="perception/robot/semantic_mapper"
SEMANTIC_PREDICTION_RELPATH="${SEMANTIC_PREDICTION_RELDIR}/semantic_prediction.py"
SEMANTIC_PREDICTION_PATH="${DROIDLET_DIR}/${SEMANTIC_PREDICTION_RELPATH}"

echo "starting semantic prediction server via $SEMANTIC_PREDICTION_PATH" 
python $SEMANTIC_PREDICTION_PATH --robot_name $ROBOT_NAME &> semseg_prediction_service.out &
timeout --foreground 20s bash -c "until python check_connected.py semantic_prediction $PYRO_IP; do sleep 0.5; done;" || true

python slam_service.py $CAMERA_NAME &> slam_service.out &
timeout --foreground 20s bash -c "until python check_connected.py slam $PYRO_IP; do sleep 0.5; done;" || true

python planning_service.py &> planning_service.out &
timeout --foreground 20s bash -c "until python check_connected.py planner $PYRO_IP; do sleep 0.5; done;" || true

python navigation_service.py $ROBOT_NAME &> navigation_service.out &
timeout --foreground 20s bash -c "until python check_connected.py navigation $PYRO_IP; do sleep 0.5; done;" || true
