#!/usr/bin/env bash
set -ex

pushd $(dirname ${BASH_SOURCE[0]})

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True

echo "Kill matching processes..."
./kill_pyro_habitat.sh

echo "Launching environment ..."

default_ip=$(hostname -I | cut -f1 -d" ")
ip=${ROBOT_IP:-$default_ip}
export LOCAL_IP=$ip
export PYRO_IP=$ip
export ROBOT_IP=$ip
echo "Binding to Host IP" $ip

export ROBOT_NAME="remoterobot"
export CAMERA_NAME="remoterobot"


python -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

echo $ip

python remote_robot.py --ip $ip $@ &
# blocking wait for server to start
timeout 1m bash -c "until python check_connected.py remoterobot $ip; do sleep 1; done;" || true
./launch_navigation.sh

popd
