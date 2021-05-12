# Copyright (c) Facebook, Inc. and its affiliates.

# starts the robot base, camera and arm 
xterm -e "roslaunch locobot_control main.launch use_base:=true use_arm:=true use_camera:=true" &

# launches the module to track AR marker on robot arm, useful for taking into account camera calibration error while exectuubg graps 
xterm -e "roslaunch locobot_calibration ar_track_alvar_calibration.launch max_new_marker_error:=0.05 marker_size:=2.8" &

load_pyrobot_env="source /home/locobot/pyenv_pyrobot_python3/bin/activate && source /home/locobot/pyrobot_catkin_ws/devel/setup.bash && export PYRO_SERIALIZER='pickle' && export PYRO_SERIALIZERS_ACCEPTED='pickle'"
ip=($(hostname -I))
echo "Host IP" ${ip[0]}
xterm -hold -e "$load_pyrobot_env; python -m Pyro4.naming -n ${ip[0]};" &
xterm -hold -e "$load_pyrobot_env; python remote_locobot.py --ip ${ip[0]};" &
echo "Launching the pyro environment on the locobot ..."
sleep 15
