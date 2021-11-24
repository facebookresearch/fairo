#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.

S3_DEST=s3://droidlet-hitl

function background_agent() (
    echo "Running craftassist agent"
    python3 /fairo/droidlet/lowlevel/minecraft/craftassist_cuberite_utils/wait_for_cuberite.py --host localhost --port 25565
    python3 /fairo/agents/craftassist/craftassist_agent.py --no_default_behavior --agent_debug_mode --enable_timeline --log_level debug --dev 1>agent.log 2>agent.log
)

echo "Installing Droidlet as a module"

cd /fairo && python3 setup.py develop && cd /

python3 /fairo/droidlet/lowlevel/minecraft/cuberite_process.py \
    --mode creative \
    --workdir . \
    --config flat_world \
    --seed 0 \
    --logging \
    --add-plugin shutdown_on_leave \
    --add-plugin shutdown_if_no_player_join \
    1>cuberite_process.log \
    2>cuberite_process.log \
    &

background_agent

# if turk_experiment_id.txt is provided, write to a turk bucket
if test -f "turk_experiment_id.txt"; then
    turk_experiment_id="$(cat turk_experiment_id.txt)"
    S3_DEST="$S3_DEST/$turk_experiment_id/interaction"
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
