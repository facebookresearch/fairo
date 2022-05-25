#!/bin/env bash
set -ex
# Runs label prop on the reexploration data.
# Usage
# ./launch_label_prop.sh <path that is output dir of ./launch_candidate_selection.sh> <name of out_dir> <setting: instance or class>
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/test_pr2/test_pr2 test_pr2 instance

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/labelprop
out_dir=/checkpoint/${USER}/jobs/reexplore/labelprop/$2

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

chmod +x run_label_prop.py
python run_label_prop.py --data_dir $data_dir --job_dir $job_dir --out_dir $out_dir --setting $3