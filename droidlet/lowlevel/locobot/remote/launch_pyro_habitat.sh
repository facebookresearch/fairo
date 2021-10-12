# Copyright (c) Facebook, Inc. and its affiliates.

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True

echo "Ensuring clean slate (kills roscore, rosmaster processes)..."
killall -9 roscore
killall -9 rosmaster
killall python
sleep 5

echo "Launching environment ..."
roscore &
source /root/pyenv_pyrobot_python3/bin/activate && source /root/pyrobot_catkin_ws/devel/setup.bash

default_ip=$(hostname -I)
ip=${LOCOBOT_IP:-$default_ip}
export LOCAL_IP=$ip
export LOCOBOT_IP=$ip
echo "Binding to Host IP" $ip

python -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

pushd $(dirname ${BASH_SOURCE[0]})
echo $ip
python remote_locobot.py --ip $ip --backend habitat &
BGPID2=$!

sleep 10

python slam_service.py &
BGPID3=$!

sleep 1

python planning_service.py &
BGPID3=$!

sleep 1

python navigation_service.py
BGPID3=$!



