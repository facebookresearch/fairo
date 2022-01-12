#!/bin/env bash
set -ex

# Usage
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3_noisy --noisy
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3

if ! source activate /private/home/apratik/miniconda3/envs/droidlet; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/miniconda3/envs/droidlet
fi

data_dir=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/reexplore/labelprop

dt=$(date '+%d-%m-%Y/%H:%M:%S');
job_dir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $job_dir
mkdir -p $job_dir
echo """"""""""""""""""""""""""""""

cd /private/home/apratik/fairo/tools/active_vision

chmod +x run_label_prop.py
python run_label_prop.py --data_dir $data_dir --job_dir $job_dir

chmod +x prep_and_run_training.py
python prep_and_run_training.py --data_dir $data_dir --job_dir $job_dir