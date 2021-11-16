#!/bin/env bash
set -ex

./kill.sh

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True
export PYRO_PICKLE_PROTOCOL_VERSION=2

default_ip=$(hostname -I | cut -f1 -d" ")
ip=${LOCOBOT_IP:-$default_ip}
echo "Binding to Host IP" $ip

python3 -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

echo $ip
python3 ./remote_hello_robot.py --ip $ip &
BGPID2=$!

python3 ./remote_hello_realsense.py --ip $ip &
BGPID3=$!

python3 ./remote_hello_saver.py --ip $ip
BGPID3=$!
