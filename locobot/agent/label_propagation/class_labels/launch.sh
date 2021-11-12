#!/bin/env bash
set -ex

if ! source activate /private/home/apratik/.conda/envs/denv3; then
    echo "source activate not working, trying conda activate"
    source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh || true
    conda activate /private/home/apratik/.conda/envs/denv3
fi

# ./launch.sh <root folder with all trajectory data> <setting specific path> <num of trajectories> <num of training runs> <slurm or local>
# Example commands to run this file
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019 apartment_0/straightline/no_noise 10 5


data_path=$1
# Base dir for all jobs
base_dir=/checkpoint/${USER}/jobs/active_vision/pipeline/$2
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
cp ../label_propagation.py $codedir/label_propagation.py
cp slurm_train.py $codedir/slurm_train.py
cp run_pipeline.py $codedir/run_pipeline.py
cp candidates.py $codedir/candidates.py

cd $codedir
chmod +x run_pipeline.py
python3.7 run_pipeline.py --data $data_path --job_folder $jobdir --num_traj $3 --num_train_samples $4 $5 --slurm

# please clean up your checkpoints before running these so as not to exceed the 8TB space limit
# cd /checkpoint/$USER/jobs/active_vision/pipeline 
# find . -name output_aug -exec rm -rf {} \;

# arthur's
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200_combined_2_sampled20 apartment_0/default/no_noise 20 3 

# soumith's
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019_sampled20 apartment_0/straightline/no_noise 20 3 --active 

# kavya's
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/noise/mul_traj_201_sampled apartment_0/straightline/noise 20 3 --active

# yuxuan's
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/noise/mul_traj_200_sampled apartment_0/default/no_noise 20 3

# anurag's
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019_sampled20 apartment_0/straightline/no_noise 2 2 --active
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200_combined_2_sampled20 apartment_0/default/no_noise 2 2

# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019 apartment_0/straightline/no_noise 2 2 --active
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200_combined_2 apartment_0/default/no_noise 5 2
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/noise/mul_traj_200_combined_2 apartment_0/default/noise 2 2
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/noise/mul_traj_200_combined_2 apartment_0/default/noise 5 3
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200 apartment_0/default/no_noise 2 2
# ./launch.sh /checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200_combined_2 apartment_0/default/no_noise 10 3
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019 apartment_0/straightline/no_noise 20 3 --active
# ./launch.sh /checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019 apartment_0/straightline/no_noise 2 2 --active
