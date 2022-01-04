#!/bin/bash 

export CONDA_SHLVL=1
export LD_LIBRARY_PATH=/root/low_cost_ws/devel/lib:/opt/ros/melodic/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CONDA_EXE=/root/miniconda3/bin/conda
export ROS_ETC_DIR=/opt/ros/melodic/etc/ros
export CONDA_PREFIX=/root/miniconda3
export NVIDIA_VISIBLE_DEVICES=all
export CONDA_PYTHON_EXE=/root/miniconda3/bin/python
export CMAKE_PREFIX_PATH=/root/low_cost_ws/devel:/opt/ros/melodic
export _CE_CONDA=
export ROS_ROOT=/opt/ros/melodic/share/ros
export ROS_MASTER_URI=http://localhost:11311
export CONDA_PROMPT_MODIFIER=(base) 
export ROS_VERSION=1
export ROS_PYTHON_VERSION=2
export CUDA_PKG_VERSION=10-1=10.1.243-1
export CUDA_VERSION=10.1.243
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,compat32,utility
export SHLVL=1
export PYTHONPATH=/root/low_cost_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:/usr/local/cython
export NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419
export ROS_PACKAGE_PATH=/root/low_cost_ws/src:/opt/ros/melodic/share
export ROSLISP_PACKAGE_DIRECTORIES=/root/low_cost_ws/devel/share/common-lisp
export PATH=/root/miniconda3/bin:/root/miniconda3/condabin:/opt/ros/melodic/bin:/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CONDA_DEFAULT_ENV=base
export PKG_CONFIG_PATH=/root/low_cost_ws/devel/lib/pkgconfig:/opt/ros/melodic/lib/pkgconfig
export ROS_DISTRO=melodic

heu=straightline #default straightline
scene=$1 # apartment_0 room_0 office_2
x=no_noise # no_noise noise
datetime=test1 #$(date +%s)
export SAVE_VIS=true #true false (when debugging)
export SLAM_SAVE_FOLDER="./data/${scene}/${heu}/${x}/${datetime}"
echo $SLAM_SAVE_FOLDER
export SCENE=$scene # changed remote locobot to use this
export HEURISTIC=$heu # changed default_behavior to use this 
export NUMTRAJ=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Using exploration heuristic ${heu}"
if [ $x == "noise" ]; then
    export NOISY_HABITAT=true
fi
# launch habitat
./droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh

#