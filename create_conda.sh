#!/usr/bin/env bash
set -ex

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate base

conda env remove -n droidlet -y

conda install mamba -c conda-forge -y

mamba create -n droidlet python=3.7 --file conda.txt --file agents/locobot/conda.txt \
      -c pytorch -c aihabitat -c open3d-admin -c conda-forge  -y

conda activate droidlet
mamba install https://anaconda.org/aihabitat/habitat-sim/0.2.1/download/linux-64/habitat-sim-0.2.1-py3.7_headless_linux_fc7fb11ccec407753a73ab810d1dbb5f57d0f9b9.tar.bz2

python setup.py develop

pip install -r agents/locobot/requirements.txt
