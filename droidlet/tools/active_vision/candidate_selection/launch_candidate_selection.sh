#!/bin/env bash
# Finds locations in simulation that the robot should respawn at to execute re-explore.
# The locations are derived from frames selected by the candidate selection heuristic at droidlet/perception/robot/active_vision/candidate_selection.py
# Usage
# ./launch_candidate_selection.sh <path to explore trajectories> <out_dir name> <number of trajectories to run for> <setting: instance or class>
# ./launch_candidate_selection.sh /checkpoint/${USER}/data_reexplore/test_pr2 test_pr2 2 instance 


set -ex

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore
mkdir -p $base_dir
dt=$(date '+%d-%m-%Y/%H:%M:%S');

out_dir=$base_dir/$2
num_traj=$3

# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/candidates

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

chmod +x find_candidates.py
python find_candidates.py --data_dir $data_dir --out_dir $out_dir --num_traj $num_traj --job_dir $job_dir --setting $4