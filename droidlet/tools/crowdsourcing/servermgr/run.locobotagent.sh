#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.

S3_DEST=s3://craftassist/turk_interactions_with_agent

function background_agent() (
    echo "Running locobot agent"
    python3 agents/locobot/locobot_agent.py --no_default_behavior
)

echo "Installing Droidlet as a module"
cd /droidlet && python3 setup.py develop && cd ..
ls

background_agent
halt
