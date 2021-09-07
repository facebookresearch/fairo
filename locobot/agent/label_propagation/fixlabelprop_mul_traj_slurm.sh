source activate /private/home/apratik/.conda/envs/locobot_env

export EXPPREFIX=_multraj_test1

# Base dir for all jobs
base_dir=/checkpoint/apratik/jobs/active_vision/label_propagation
mkdir -p $base_dir

# Create a directory for each job 
# make a job-specific dir
dt=$(date '+%d-%m-%Y_%H:%M:%S');
echo $dt

jobdir=$base_dir/"${1}_${EXPPREFIX}_${dt}"
echo $jobdir
mkdir -p $jobdir

# pass jobdir, samples to .py
# Example: ./launch.sh 1_default_gt10px 1000, where gt frames is fixed but propagation length is not, and we want 1000 samples
cp label_propagation.py $jobdir/label_propagation.py
cd $jobdir
echo $jobdir
chmod +x label_propagation.py

# ./slurm_train.py --job_folder $jobdir --samples $2


# export SCENE_ROOTD=/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200
# export SCENE_ROOTD=/checkpoint/apratik/data/data/apartment_0/default/noise/mul_traj_200
export SCENE_ROOTD=/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/mul_traj_201
# echo $SCENE_ROOTD 

# END=100
# for i in $(seq $END); 
#     do echo $SCENE_ROOTD/$i
#     for gt in 5 10 15 20 25
#         do
#         export OUTDIR=$SCENE_ROOTD/$i/pred_label_gt${gt}p2fix$EXPPREFIX
#         echo $OUTDIR
#         python label_propagation.py --scene_path $SCENE_ROOTD/$i --gtframes $gt --propogation_step 2 --out_dir $OUTDIR
#     done    
# done

# Test on one scene
export i=$2
echo $SCENE_ROOTD/$i
for gt in 5 10 15 20 25
    do
    export OUTDIR=$SCENE_ROOTD/$i/pred_label_gt${gt}p2fix$EXPPREFIX
    echo $OUTDIR
    python label_propagation.py --scene_path $SCENE_ROOTD/$i --gtframes $gt --propogation_step 2 --out_dir $OUTDIR --job_folder $jobdir
done    
# # done
