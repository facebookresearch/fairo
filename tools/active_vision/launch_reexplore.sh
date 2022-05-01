#!/bin/env bash
set -ex
# Launches re-exploraation from locations output by ./launch_candidate selection.
# Usage
# ./launch_reexplore <path that is output dir of ./launch_candidate_selection.sh> <setting for noise>
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/hail_mary2/baselinev3
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/fifty_422/baselinev3
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/collection2_426/baseline
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/collection2_preemp_test/baseline

# git checkout ap/reex_chkpt
# cd tools/active_vision
# chmod +x launch_reexplore.sh
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/av300/av300_simple
# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/av300_noise/av300_noise_simple --noise


# /checkpoint/apratik/jobs/reexplore/av300/av300_simple/1/instance/5/candidate_selection_visuals/00037.jpg

# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/av300/av300_simple

# ./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/largerun_noisy/baselinev3_noisy --noise

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

cd /private/home/apratik/fairo/tools/active_vision

chmod +x reexplore.py
python reexplore.py --data_dir $data_dir --job_dir $job_dir $2