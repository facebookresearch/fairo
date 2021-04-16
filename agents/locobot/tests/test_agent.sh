#!/bin/bash
set -ex

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $SCRIPT_DIR

pushd ..
python locobot_agent.py 2>&1 >/dev/null &
BGPID=$!
popd

sleep 45 # wait for the agent to fully start (including downloading of models

python test_agent.py

wait $BGPID
STATUS=$?

exit $STATUS
