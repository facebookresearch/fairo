#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

start=`date +%s`
echo "Ensure Habitat is running and LOCOBOT_IP is set."
cd $(dirname $0)
coverage run --source . -m unittest discover -p 'test*.py'
status=$?
coverage report
end=`date +%s`
runtime=$((end-start))
echo "Runtime $runtime seconds"
exit $status
