# Copyright (c) Facebook, Inc. and its affiliates.

export PATH="$HOME/realsense_install/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/realsense_install/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PYTHONPATH:$HOME/realsense_install/lib/python3.6"

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

default_ip=$(ifconfig wlan0| grep 'inet '| awk '{ print $2}')
ip=${LOCOBOT_IP:-$default_ip}
echo "Binding to Host IP" $ip

python3 -m Pyro4.naming -n $ip &
BGPID=$!
sleep 4

pushd $(dirname ${BASH_SOURCE[0]})
echo $ip
python3 remote_dji.py --host $ip
BGPID2=$!

