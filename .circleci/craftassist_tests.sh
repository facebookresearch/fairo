#!/bin/bash
set -ex

script_dir=$(dirname $0)
SHARED_PATH=/shared

pushd $script_dir/../

pytest --cov-report=xml:$SHARED_PATH/craftassist_tests.xml --cov=droidlet droidlet/ --disable-pytest-warnings \
       --ignore droidlet/memory/robot/ \
       --ignore droidlet/perception/robot/ \
       --ignore droidlet/interpreter/robot/ \
       --ignore droidlet/lowlevel/locobot/
