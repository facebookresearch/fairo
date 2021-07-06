#!/bin/bash -x
# Copyright (c) Facebook, Inc. and its affiliates.

S3_DEST=s3://craftassist/turk_interactions_with_agent

function background_agent() (
    echo "Running locobot agent"
    export LOCOBOT_IP="172.17.0.2"
    ls
    conda init bash
    conda activate droidlet_env
    python3 agents/locobot/locobot_agent.py --no_default_behavior
)

echo "Installing Droidlet as a module"
ls
cd droidlet
# conda create -n droidlet_env python=3.7 \
#    pytorch==1.7.1 torchvision==0.8.2 \
#    cudatoolkit=11.0 -c pytorch
conda activate droidlet_env
# pip install -r \
#     agents/locobot/requirements.txt
cd /droidlet && python3 setup.py develop && cd ..
ls

background_agent
halt
