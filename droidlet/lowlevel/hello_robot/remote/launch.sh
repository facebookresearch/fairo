# Copyright (c) Facebook, Inc. and its affiliates.

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SOCK_REUSE=True
export PYRO_PICKLE_PROTOCOL_VERSION=2

default_ip=172.20.4.241 #$(hostname -I)
ip=${LOCOBOT_IP:-$default_ip}
echo "Binding to Host IP" $ip

python3 -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

echo $ip
python3 ./remote_hello_robot.py --ip $ip
BGPID2=$!
