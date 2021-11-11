#!/bin/env bash
set -ex

if ! source activate /private/home/apratik/.conda/envs/denv3; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/.conda/envs/denv3
fi

# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/active_vision/pipeline/sanity_check0
mkdir -p $base_dir
dt=$(date '+%d-%m-%Y/%H:%M:%S');

jobdir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $jobdir
mkdir -p $jobdir
echo """"""""""""""""""""""""""""""

codedir=$jobdir/code
mkdir -p $codedir

cp run_sanity_check0.py $codedir/run_sanity_check0.py

cd $codedir
chmod +x run_sanity_check0.py
python3.7 run_sanity_check0.py --job_folder $jobdir --slurm 

# ./launch_sanity.sh