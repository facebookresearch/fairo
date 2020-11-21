#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.



S3_DEST=s3://craftassist/workdirs/humanhuman

python3 /minecraft/python/cuberite_process.py \
    --mode creative \
    --workdir . \
    --config flat_world \
    --seed 0 \
    --logging \
    --add-plugin shutdown_on_leave \
    1>cuberite_process.log \
    2>cuberite_process.log


TARBALL=workdir.$(date '+%Y-%m-%d-%H:%M:%S').$(hostname).tar.gz
tar czf $TARBALL . --force-local

if [ -z "$CRAFTASSIST_NO_UPLOAD" ]; then
    aws s3 cp $TARBALL $S3_DEST/$TARBALL
fi

halt
