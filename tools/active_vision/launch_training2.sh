#!/bin/env bash
set -ex
# Runs coco-ization and training on label propagated datasets
# Usage
# ./launch_training.sh <path that is output dir of ./launch_candidate_selection.sh> <setting for noise>
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/largerun1/baselinev3
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/smoothed_infocus_test7
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/collection2_426
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/av300
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/av300_pt2/  
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/av_sanity50
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3_noisy 
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/labelprop/av_sm_noise_50 instance
# ./launch_training2.sh /checkpoint/apratik/jobs/reexplore/train_data_solo3/train_data_solo2 instance

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/training

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

chmod +x prep_and_run_training.py
python prep_and_run_training2.py --data_dir $data_dir --job_dir $job_dir --num_train_samples 2 --setting $2