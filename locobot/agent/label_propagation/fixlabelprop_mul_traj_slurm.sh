source activate /private/home/apratik/.conda/envs/locobot_env

export EXPPREFIX=_multraj

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
export SCENE_ROOTD=/checkpoint/apratik/data/data/apartment_0/default/noise/mul_traj_200
# export SCENE_ROOTD=/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/mul_traj_201
# export SCENE_ROOTD=/checkpoint/apratik/data_devfair0187/apartment_0/straightline/noise/mul_traj_201
# echo $SCENE_ROOTD 

# for i in 3 5 8 15 16 23 27 29 34 35 39 44 53 57 61 63 64 65 67 76 86  # active no noise
# for i in 7 22 40 49 73 79 92 # active noise
# for i in 123 66 124 140 44 6 60 146 65 11 82 131 87 45 59 114 81 184 4 192 #non active no noise
for i in 176 135 182 161 179 105 154 43 168 89 172 72 191 116 177 111 186 169 28 113 # non active noise
    do echo $SCENE_ROOTD/$i
    for gt in 5 10 15 20 25
        do
        export OUTDIR=$SCENE_ROOTD/$i/pred_label_gt${gt}p2fix${EXPPREFIX}_${dt}
        echo $OUTDIR
        python label_propagation.py --scene_path $SCENE_ROOTD/$i --gtframes $gt --propogation_step 2 --out_dir $OUTDIR --job_folder $jobdir
    done    
done

# # Test on one scene
# export i=$2
# echo $SCENE_ROOTD/$i
# for gt in 5 10 15 20 25
#     do
#     export OUTDIR=$SCENE_ROOTD/$i/pred_label_gt${gt}p2fix${EXPPREFIX}_${dt}
#     echo $OUTDIR
#     python label_propagation.py --scene_path $SCENE_ROOTD/$i --gtframes $gt --propogation_step 2 --out_dir $OUTDIR --job_folder $jobdir
# done    
# # done
