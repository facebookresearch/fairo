#!/bin/env bash
export NOISY_HABITAT=$3
export ADD_HUMANS=False
source activate /private/home/${USER}/miniconda3/envs/droidlet
../../droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh 127.0.0.1 --scene_path /checkpoint/apratik/replica/apartment_0/habitat/mesh_semantic.ply

export LOCOBOT_IP=127.0.0.1
export SAVE_EXPLORATION=True
export DATA_PATH=test
export VISUALIZE_EXAMINE=True
source activate /private/home/${USER}/miniconda3/envs/droidlet
python ../../agents/locobot/locobot_agent.py --data_store_path $1 --default_behavior reexplore --reexplore_json $2