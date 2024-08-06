# Getting the Remote service up and running

On the Hello Robot Stretch, run the following from the directory where the readme exists

## 1. Install venv and requirements and hector slam

```bash
python3 -m venv droidlet
. droidlet/bin/activate
pip install -r requirements.txt

sudo apt install ros-noetic-hector-slam -y
```

## 2. Install droidlet in develop mode

```bash
pushd ../../../../
python setup.py develop
popd
```

## 3. Start the services

### Without ROS (i.e. directly using base API)

```bash
# export LOCOBOT_IP=[your network ip]
# example when using tailscale:
# export LOCOBOT_IP=$(tailscale ip --4)

./launch.sh
```

### With ROS (i.e. better odometry using hector_slam, etc.)

```bash
# in fresh terminal
deactivate
./launch_ros.sh

# in another terminal
. droidlet/bin/activate

# export LOCOBOT_IP=[your network ip]
# example when using tailscale:
# export LOCOBOT_IP=$(tailscale ip --4)

./launch.sh --ros
```