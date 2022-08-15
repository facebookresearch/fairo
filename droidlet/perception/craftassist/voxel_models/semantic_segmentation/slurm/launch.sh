#!/bin/bash

# Usage: ./slurm/launch.sh job_name partition ngpus constraint nodes arg1 arg2 ...
name=$1
partition=$2
ngpus=$3
constraint=$4
nodes=$5
args=${@:6}

base_dir=/checkpoint/$USER/jobs/hitl_vision/$name
mkdir -p $base_dir

# backup the code
codedir=$base_dir/code/
if [ -d "$codedir" ]; then
  read -r -p "The code already exists. Overwrite it? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/Workspace/hitl/fairo/droidlet/perception/craftassist/voxel_models/semantic_segmentation/ $codedir
    # log the git state
    echo "$args" > $base_dir/args.txt
    git log|head -6 > $base_dir/git.txt
    echo -e "\n\n" >> $base_dir/git.txt
    git status >> $base_dir/git.txt
    echo -e "\n\n" >> $base_dir/git.txt
    git diff >> $base_dir/git.txt
  fi
else
  echo "Copying code into slurm code dir"
  rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/Workspace/hitl/fairo/droidlet/perception/craftassist/voxel_models/semantic_segmentation/ $codedir
  # log the git state
  echo "$args" > $base_dir/args.txt
  git log|head -6 > $base_dir/git.txt
  echo -e "\n\n" >> $base_dir/git.txt
  git status >> $base_dir/git.txt
  echo -e "\n\n" >> $base_dir/git.txt
  git diff >> $base_dir/git.txt
fi

cd $codedir/
host=$(hostname)
export PYTHONPATH=$codedir:$PYTHONPATH
if [[ "$partition" = "local" ]]; then
  echo running locally $name
  python main.py $args --checkpoint $base_dir/model.pt --plot --plot-name $name
else
  python slurm/submit.py --name $name --folder $base_dir --partition $partition --ngpu $ngpus --constraint "$constraint" --nodes $nodes \
    --args "$args --save_model $base_dir/model.pt"
fi