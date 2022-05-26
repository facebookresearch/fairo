#!/bin/env bash
set -ex
# Launches re-exploraation from locations output by ./launch_candidate selection.
# Usage
# ./launch_reexplore <path that is output dir of ./launch_candidate_selection.sh> <setting: instance or class> <setting for noise>
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/test_pr2/test_pr2 instance
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/test_pr2/test_pr2 instance --noise

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/recollect

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

chmod +x reexplore.py
chmod +x reexplore.sh
python reexplore.py --data_dir $data_dir --job_dir $job_dir --setting $2 $3 