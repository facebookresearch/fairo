#!/bin/env bash
set -ex
# Runs coco-ization and training on label propagated datasets
# Usage
# ./launch_training.sh <path that is output dir of ./launch_candidate_selection.sh> <setting for noise>
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/respawnv4/baselinev3
# ./launch_training.sh /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3_noisy --noisy

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

cd /private/home/apratik/fairo/tools/active_vision

chmod +x prep_and_run_training.py
python prep_and_run_training.py --data_dir $data_dir --job_dir $job_dir