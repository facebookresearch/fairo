# Copyright (c) Facebook, Inc. and its affiliates.

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True

echo "Ensuring clean slate (kills roscore, rosmaster processes)..."
ps -elf|grep remote_locobot.py | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1
ps -elf|grep Pyro4.naming | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1
sleep 1

echo "Launching environment ..."

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

sleep 45

./launch_navigation.sh &
