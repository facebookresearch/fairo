#!/bin/bash -xe

conda install -y conda-build
conda build -c conda-forge --no-anaconda-upload .

conda install -y anaconda-client
#export ANACONDA_PASSWORD="5T!%NUjCZy529@d"
#export ANACONDA_USER=tingfan
#anaconda login --username $ANACONDA_USER --password $ANACONDA_PASSWORD

TARGETS=$(conda build --no-anaconda-upload . --output)
echo "Uploading $TARGETS, Ctrl-C to stop"
sleep 5
anaconda upload --user fair-robotics --force $TARGETS
