#!/bin/env bash
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

out_dir=$base_dir/respawnv2
# job_dir=$base_dir/$dt
# echo """"""""""""""""""""""""""""""
# echo Job Directory $job_dir
# mkdir -p $job_dir
# echo """"""""""""""""""""""""""""""

cd /private/home/apratik/fairo/tools/active_vision
chmod +x find_respawn_loc.py
python3.7 find_respawn_loc.py --data_dir $data_dir --out_dir $out_dir --slurm 
