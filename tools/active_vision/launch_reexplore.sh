#!/bin/env bash
set -ex

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

# /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3_noisy
# /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

cd /private/home/apratik/fairo/tools/active_vision

chmod +x reexplore.py
python reexplore.py --data_dir $data_dir --job_dir $job_dir #--noise