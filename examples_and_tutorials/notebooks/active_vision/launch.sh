base_dir=/checkpoint/apratik/jobs/active_vision
mkdir -p $base_dir

# backup the code
codedir=$base_dir/code/
mkdir -p $codedir

cp slurm_train.py $codedir/slurm_train_default1k_gtfix_final.py
cd $codedir
echo $PWD
chmod +x slurm_train_default1k_gtfix_final.py
./slurm_train_default1k_gtfix_final.py