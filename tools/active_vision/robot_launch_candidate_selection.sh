#!/usr/bin/env bash
# Finds locations in simulation that the robot should respawn at to execute re-explore.
# The locations are derived from frames selected by the candidate selection heuristic at droidlet/perception/robot/active_vision/candidate_selection.py
# Usage

# ./robot_launch_candidate_selection.sh  ${HOME}/explore_data/default/0 ${HOME}/explore_data/default/0/reexplore

set -ex

if ! source activate /home/locobotm/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /home/locobotm/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
out_dir=$2
mkdir -p out_dir


chmod +x find_respawn_loc.py
python find_respawn_loc.py --data_dir $data_dir --out_dir $out_dir --mode robot