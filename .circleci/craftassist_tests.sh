#!/bin/bash
set -ex

script_dir=$(dirname $0)
SHARED_PATH=/shared

pushd $script_dir/../

pytest --cov-report=xml:$SHARED_PATH/test_full_craftassist_agent.xml --cov=$COV_RELATIVE agents/craftassist/tests/ --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/droidlet_craftassist_tests.xml --cov=droidlet droidlet/ --disable-pytest-warnings \
       --ignore droidlet/memory/robot/ \
       --ignore droidlet/perception/robot/ \
       --ignore droidlet/interpreter/robot/ \
       --ignore droidlet/lowlevel/locobot/

if [ ! -f coverage.xml ];
   pip install coverage
   coverage xml -i
fi
cp coverage.xml /shared
