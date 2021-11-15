#!/bin/env bash
set -ex

if ! source activate /private/home/apratik/.conda/envs/denv3; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/.conda/envs/denv3
fi

# ./launch.sh <root folder with all trajectory data> <setting specific path> <num of training runs> <slurm or local>
# Example commands to run this file
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/instance_detection_ids_allinone apartment_0/straightline/no_noise 1 --slurm


data_path=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/active_vision/pipeline/instance_det/$2
mkdir -p $base_dir
dt=$(date '+%d-%m-%Y/%H:%M:%S');

jobdir=$base_dir/$dt
echo """"""""""""""""""""""""""""""
echo Job Directory $jobdir
mkdir -p $jobdir
echo """"""""""""""""""""""""""""""

codedir=$jobdir/code
mkdir -p $codedir

cp coco.py $codedir/coco.py
cp label_propagation.py $codedir/label_propagation.py
cp slurm_train.py $codedir/slurm_train.py
cp run_pipeline.py $codedir/run_pipeline.py
cp candidates.py $codedir/candidates.py

echo $5

cd $codedir
chmod +x run_pipeline.py
python3.7 run_pipeline.py --data $data_path --job_folder $jobdir --num_train_samples $3 --comment $4 $5

# play book to run the pipeline on Hello
# pick frames to label, one per instance for the train set where the instance.  
# convert coco format labels into seg format (each pixel is the instance id, same dim as the image)
# set labeled image ids to src_img_ids in run_pipeline.py
# plug and play ./launch.sh <the root folder with rgb, depth, seg, data.json> <scene>/<heu>/real_world num_train_sample --slurm


# please clean up your checkpoints before running these so as not to exceed the 8TB space limit
# cd /checkpoint/$USER/jobs/active_vision/pipeline 
# find . -name output_droid -exec rm -rf {} \;

# anurag's
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/instance_detection_ids_allinone_auto apartment_0/straightline/no_noise 3 --slurm
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/instance_detection_ids_allinone_3 apartment_0/default/no_noise 3 --slurm 
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/instance_detection_ids_allinone_auto apartment_0/straightline/no_noise 3 --slurm
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/instance_detection_ids_allinone_3 apartment_0/default/no_noise 3 --slurm 

# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1/active apartment_0/straightline/no_noise 1 --slurm "test commment"
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1/baseline apartment_0/default/no_noise 1 --slurm "only prop candidate instance label"

# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/instance_traj_man apartment_0/straightline/no_noise 1 --slurm
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled2/baseline apartment_0/default/no_noise 1 --slurm

# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled3/active apartment_0/straightline/no_noise 1 --slurm
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled3/baseline apartment_0/default/no_noise 1 --slurm


# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1115_nodownsamplingbaseline/active apartment_0/straightline/no_noise 1 --slurm prop_all_no_downsampling_baseline
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1115_nodownsamplingbaseline/baseline apartment_0/default/no_noise 1 --slurm prop_all_no_downsampling_baseline

# /checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1114