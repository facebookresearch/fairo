heu=straightline #default straightline
scene=apartment_0 # apartment_0 room_0 office_2
x=noise # no_noise noise
datetime=mul_traj_200 #$(date +%s)
export SAVE_VIS=false #true false (when debugging)
export SLAM_SAVE_FOLDER="/checkpoint/apratik/data/${scene}/${heu}/${x}_${datetime}"
echo $SLAM_SAVE_FOLDER
export SCENE=$scene # changed remote locobot to use this
export HEURISTIC=$heu # changed default_behavior to use this 
echo "Using exploration heuristic ${heu}"
if [ $x == "noise" ]; then
    export NOISY_HABITAT=true
fi

# launch habitat
./droidlet/lowlevel/locobot/remote/launch_pyro_habitat.sh