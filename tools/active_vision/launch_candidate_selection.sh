#!/bin/env bash
# Finds locations in simulation that the robot should respawn at to execute re-explore.
# The locations are derived from frames selected by the candidate selection heuristic at droidlet/perception/robot/active_vision/candidate_selection.py
# Usage
# ./launch_candidate_selection.sh <path to explore trajectories> <out_dir name> <number of trajectories to run for>

# No noise experiments
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/baselinev3 fifty_422 100
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av300_pt2 300
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av300_sanity50 50
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2 av_sm_class_50 50 class pfix

# Noisy Experiments
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_noise_pt2/ av300_noise_pt2 300
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_noise_pt2 av_sm_noise_50 50 instance

# Ablations
# ./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2 av_ablation_instance_50 50 instance gtfix


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

chmod +x find_respawn_loc.py
python3.7 find_respawn_loc.py --data_dir $data_dir --out_dir $out_dir --num_traj $num_traj --job_dir $job_dir --setting $4 --gt_or_p_fix $5