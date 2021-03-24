#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.



S3_DEST=s3://craftassist/turk_interactions_with_agent/$TIMESTAMP

function background_agent() (
    python3 /droidlet/craftassist/wait_for_cuberite.py --host localhost --port 25565
    python3 /droidlet/craftassist/agent/craftassist_agent.py 1>agent.log 2>agent.log
)

background_agent &


python3 /droidlet/craftassist/cuberite_process.py \
    --mode creative \
    --workdir . \
    --config flat_world \
    --seed 0 \
    --logging \
    --add-plugin shutdown_on_leave \
    --add-plugin shutdown_if_no_player_join \
    1>cuberite_process.log \
    2>cuberite_process.log


TARBALL=logs.tar.gz
# Only upload the logs and CSV files
find -name "*.log" -o -name "*.csv"|tar czf $TARBALL --force-local -T -

if [ -z "$CRAFTASSIST_NO_UPLOAD" ]; then
    # expects $AWS_ACCESS_KEY_ID and $AWS_SECRET_ACCESS_KEY to exist
    aws s3 cp $TARBALL $S3_DEST/$TARBALL
fi

halt
