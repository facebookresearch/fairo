# Base dir for all jobs
base_dir=/checkpoint/apratik/jobs/active_vision
mkdir -p $base_dir

# Create a directory for each job 
# make a job-specific dir in <scenario_id>_<heuristic_gtxpy_samplesy>_<datetime>
# refer to lab notebook (https://fb.quip.com/rGFpAsFEZa47#feUACAzNUOy) for scenario_id 
jobdir=$1
echo $jobdir
dt=$(date '+%d-%m-%Y_%H:%M:%S');
echo $dt

jobdir=$base_dir/"${1}_samples${2}_${dt}"
echo $jobdir
mkdir -p $jobdir

# pass jobdir, samples to .py
# Example: ./launch.sh 1_default_gt10px 1000, where gt frames is fixed but propagation length is not, and we want 1000 samples
cp slurm_train.py $jobdir/slurm_train.py
cd $jobdir
chmod +x slurm_train.py
./slurm_train.py --job_folder $jobdir --samples $2