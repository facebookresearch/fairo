#!/bin/env bash
set -ex
# Runs label prop on the reexploration data.
# Usage
# ./launch_label_prop.sh <path that is output dir of ./launch_candidate_selection.sh> <setting for noise>
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/hail_mary2/baselinev3
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/fifty_422/baselinev3
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/collection2_preemp_test2/baseline collection2_preemp_test2
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/respawnv1/baselinev3_noisy --noisy
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/av300_noise/av300_noise_simple av300_noise
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/av300_sanity50/av300_pt2 av_sanity50
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/av_sm_noise_50/av300_noise_pt2 av_sm_noise_50 instance 
# ./launch_label_prop.sh /checkpoint/apratik/jobs/reexplore/av_sm_class2_50/av300_pt2 av_sm_class2_50 class pfix

# soumith's
# ./launch_label_prop.sh /checkpoint/soumith/jobs/reexplore/av_sm/av300_pt2 av_sm_noise_50 instance 

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
python run_label_prop.py --data_dir $data_dir --job_dir $job_dir --out_dir $out_dir --setting $3 --gt_or_p_fix $4