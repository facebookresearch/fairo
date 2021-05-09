#!/bin/bash
set -ex

echo "Downloading datasets, models ..."
yes | tools/data_scripts/try_download.sh locobot &
wait
echo "Done!"

source /opt/ros/melodic/setup.bash
export ORBSLAM2_LIBRARY_PATH=/root/low_cost_ws/src/pyrobot/robots/LoCoBot/install/../thirdparty/ORB_SLAM2
source /root/low_cost_ws/devel/setup.bash
source /root/pyenv_pyrobot_python3/bin/activate && source /root/pyrobot_catkin_ws/devel/setup.bash

roscore &

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

LOCOBOT_IP=127.0.0.1 
SHARED_PATH=/shared

python -m Pyro4.naming -n $LOCOBOT_IP &
sleep 10

python locobot/robot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
python locobot/test/smoke_test.py
kill -9 $BGPID
sleep 5

python locobot/robot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
pushd locobot/test
pytest --cov-report=xml:$SHARED_PATH/test_habitat.xml --cov=../ test_habitat.py --disable-pytest-warnings
popd
kill -9 $BGPID
sleep 5

COV_RELATIVE=locobot
python locobot/robot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
deactivate
source activate /root/miniconda3/envs/droidlet_env
pip install -r locobot/requirements.txt
python setup.py develop

pytest --cov-report=xml:$SHARED_PATH/test_mover.xml --cov=$COV_RELATIVE locobot/test/test_mover.py --disable-pytest-warnings
kill -9 $BGPID
sleep 5

deactivate
source /root/pyenv_pyrobot_python3/bin/activate
python locobot/robot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
deactivate
source activate /root/miniconda3/envs/droidlet_env

# pytest --cov-report=xml:$SHARED_PATH/test_locobot_agent.xml --cov=$COV_RELATIVE locobot/test/test_locobot_agent.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_handlers.xml --cov=$COV_RELATIVE locobot/test/test_handlers.py --disable-pytest-warnings

kill -9 $BGPID
pytest --cov-report=xml:$SHARED_PATH/test_memory.xml --cov=$COV_RELATIVE locobot/test/test_memory.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_interpreter_mock.xml --cov=$COV_RELATIVE locobot/test/test_interpreter_mock.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_utils.xml --cov=$COV_RELATIVE locobot/test/test_utils.py --disable-pytest-warnings
