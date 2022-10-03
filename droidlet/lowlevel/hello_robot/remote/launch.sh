#!/bin/env bash
set -ex

./kill.sh

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True
export PYRO_PICKLE_PROTOCOL_VERSION=2

default_ip=$(hostname -I | cut -f1 -d" ")
ip=${LOCOBOT_IP:-$default_ip}
export LOCAL_IP=$ip
export PYRO_IP=$ip
export LOCOBOT_IP=$ip
echo "Binding to Host IP" $ip

python3 -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

export ROBOT_NAME="hello_robot"
export CAMERA_NAME="hello_realsense"

echo $ip
python3 ./remote_hello_robot.py --ip $ip &
timeout 20s bash -c "until python check_connected.py hello_robot $ip; do sleep 0.5; done;" || true

python3 ./remote_hello_realsense.py --ip $ip &
timeout 20s bash -c "until python check_connected.py hello_realsense $ip; do sleep 0.5; done;" || true

python3 ./remote_hello_saver.py --ip $ip &
timeout 10s bash -c "until python check_connected.py hello_data_logger $ip; do sleep 0.5; done;" || true

./launch_navigation.sh
