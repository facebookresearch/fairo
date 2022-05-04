#!/bin/env bash
# Finds locations in simulation that the robot should respawn at to execute re-explore.
# The locations are derived from frames selected by the candidate selection heuristic at droidlet/perception/robot/active_vision/candidate_selection.py
# Usage
# ./launch_candidate_selection.sh <path to explore trajectories> <out_dir name> <number of trajectories to run for>
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/baselinev3 fifty_422 100
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av300_pt2 300
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_noise_pt2/ av300_noise_pt2 300

set -ex

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /home/hello/anaconda3/envs/droidlet_robo
fi

data_dir=$1
# Base dir for all jobs
out_dir=$2
mkdir -p out_dir


chmod +x find_respawn_loc.py
python find_respawn_loc.py --data_dir $data_dir --out_dir $out_dir --mode robot