#!/bin/bash
set -ex

script_dir=$(dirname $0)
SHARED_PATH=/shared

echo "Downloading datasets, models ..."
yes | python3 tools/data_scripts/try_download.py --agent_name craftassist &
wait
echo "Done!"

pushd $script_dir/../

pytest --cov-report=xml:$SHARED_PATH/test_full_craftassist_agent.xml --cov=agents --cov=droidlet agents/craftassist/tests/ --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/droidlet_craftassist_tests.xml --cov=droidlet droidlet/ --disable-pytest-warnings \
       --ignore droidlet/memory/robot/ \
       --ignore droidlet/perception/robot/ \
       --ignore droidlet/interpreter/robot/ \
       --ignore droidlet/lowlevel/locobot/
