base_dir=/checkpoint/apratik/jobs/active_vision
mkdir -p $base_dir

# backup the code
codedir=$base_dir/code/
mkdir -p $codedir

cp slurm_train.py $codedir
cd $codedir
echo $PWD
chmod +x slurm_train.py
./slurm_train.py