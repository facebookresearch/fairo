#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.

S3_DEST=s3://craftassist/turk_interactions_with_agent

function background_agent() (
    echo "Running locobot agent"
    # python3 /droidlet/droidlet/lowlevel/minecraft/craftassist_cuberite_utils/wait_for_cuberite.py --host localhost --port 25565
    export LOCOBOT_IP="172.17.0.2"
    python3 /droidlet/agents/locobot/locobot_agent.py --no_default_behavior
)

echo "Installing Droidlet as a module"

docker run --gpus all -it --rm --ipc=host -v $(pwd):/remote -w /remote theh1ghwayman/locobot-assistant:5.0 bash
roscore &
load_pyrobot_env
cd locobot/robot
./launch_pyro_habitat.sh

background_agent

# if turk_experiment_id.txt is provided, write to a turk bucket
if test -f "turk_experiment_id.txt"; then
    turk_experiment_id="$(cat turk_experiment_id.txt)"
    S3_DEST="$S3_DEST/turk/$turk_experiment_id"
fi
S3_DEST="$S3_DEST/$TIMESTAMP"

TARBALL=logs.tar.gz
# Only upload the logs and CSV files
find -name "*.log" -o -name "*.csv" -o -name "job_metadata.json" -o -name "interaction_loggings.json"|tar czf $TARBALL --force-local -T -

if [ -z "$CRAFTASSIST_NO_UPLOAD" ]; then
    echo "Uploading data to S3"
    # expects $AWS_ACCESS_KEY_ID and $AWS_SECRET_ACCESS_KEY to exist
    aws s3 cp $TARBALL $S3_DEST/$TARBALL
fi

halt
