#!/bin/env bash
set -ex

source ~/miniconda3/etc/profile.d/conda.sh
conda activate droidlet

echo "Downloading datasets, models ..."
yes | tools/data_scripts/try_download.sh locobot &
wait
echo "Done!"


export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

export LOCOBOT_IP=127.0.0.1
SHARED_PATH=/shared

python -m Pyro4.naming -n $LOCOBOT_IP &
sleep 3

droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh
python droidlet/lowlevel/locobot/tests/smoke_test.py

droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh
pushd droidlet/lowlevel/locobot/tests/
pytest --cov-report=xml:$SHARED_PATH/test_habitat.xml --cov=../ test_habitat.py --disable-pytest-warnings
popd

droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh
pip install -r agents/locobot/requirements.txt
python setup.py develop

pytest --cov-report=xml:$SHARED_PATH/test_mover.xml --cov=droidlet droidlet/lowlevel/locobot/tests/test_mover.py --disable-pytest-warnings


droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh
pytest --cov-report=xml:$SHARED_PATH/test_handlers.xml --cov=droidlet droidlet/perception/robot/tests/test_perception.py --disable-pytest-warnings


pytest --cov-report=xml:$SHARED_PATH/test_memory.xml --cov=agents --cov=droidlet agents/locobot/tests/test_memory.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_interpreter_mock.xml --cov=agents --cov=droidlet agents/locobot/tests/test_interpreter_mock.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_memory_low_level.xml --cov=droidlet droidlet/memory/robot/tests/test_low_level_memory.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_utils.xml --cov=droidlet droidlet/lowlevel/locobot/tests/test_utils.py --disable-pytest-warnings


droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh
./agents/locobot/tests/test_agent.sh
