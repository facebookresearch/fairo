#!/bin/bash
set -ex

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $SCRIPT_DIR

TEST_TIMEOUT=120s

pushd ..
timeout -k $TEST_TIMEOUT $TEST_TIMEOUT \
	python locobot_agent.py &
BGPID=$!
popd

python test_agent.py
STATUS1=$?
if [[ $STATUS1 -gt 0 ]]; then
    echo "test_agent.py returned with non-zero error code."
    kill -9 $BGPID
    exit STATUS1
else
    wait $BGPID
    STATUS2=$?
    if [[ $STATUS2 -gt 0 ]]; then
	echo "Failed to shutdown locobot_agent.py cleanly."
	exit $STATUS2
    fi
fi


