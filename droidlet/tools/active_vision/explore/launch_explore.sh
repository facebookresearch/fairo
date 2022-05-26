#!/bin/env bash
set -ex
# Launches data collection on the agent, using random exploration 
# FIRST, MAKE THE BASH SCRIPT IS EXECUTABLE
# chmod +x launch_explore.sh
# ALSO THE REPO SHOULD BE IN YOUR DIRECTORY ACCESSIBLE FROM THE CLUSTER

# To launch a test job that collects 2 trajectories
# ./launch_explore.sh /checkpoint/${USER}/data_reexplore/test_pr 2

# To launch a test job that collects 5 trajectories in noisy setting 
# ./launch_explore.sh /checkpoint/${USER}/data_reexplore/test_slurm_collect2_noise 5 --noise

# To launch a test job that collects 300 trajectories 
# ./launch_explore.sh /checkpoint/${USER}/data_reexplore/av300_pt2 300

# To launch a test job that collects 300 trajectories in noisy setting
# ./launch_explore.sh /checkpoint/${USER}/data_reexplore/av300_noise_pt2 300 --noise


if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
num_traj=$2
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/collect

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

chmod +x explore.py
chmod +x explore.sh
python explore.py --data_dir $data_dir --job_dir $job_dir --num_traj $num_traj $3