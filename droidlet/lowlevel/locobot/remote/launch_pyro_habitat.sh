# Copyright (c) Facebook, Inc. and its affiliates.

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

default_ip=$(hostname -I)
ip=${LOCOBOT_IP:-$default_ip}
echo "Binding to Host IP" $ip

python -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

pushd $(dirname ${BASH_SOURCE[0]})
echo $ip
python remote_locobot.py --ip $ip --backend habitat
BGPID2=$!
