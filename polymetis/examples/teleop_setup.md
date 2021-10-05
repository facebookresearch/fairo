# Teleoperation Setup Guide

### Step 1: Install Polymetis at the dev branch locally
```bash
git clone --recurse-submodules git@github.com:facebookresearch/fairo
cd fairo
git checkout austinw/oculus_teleop
```
Installation instructions can be found [here](https://polymetis-docs.github.io/installation.html#from-source).

This will also create a new conda env for everything else to be installed into.

### Step 2: Setup Oculus

Install oculus-reader:
```bash
git clone git@github.com:rail-berkeley/oculus_reader
cd oculus_reader
pip install .
```

Then, follow instructions in the [README](https://github.com/rail-berkeley/oculus_reader/blob/main/README.md) to set up the Oculus headset.

Note: If `adb devices` does not list the Oculus device or has the error `no permissions (user in plugdev group; are your udev rules wrong?);`, restart adb with sudo access:
```bash
adb kill-server
sudo adb start-server
```

To prevent the headset from going to sleep, go to Settings --> Device --> Auto Sleep Headset and change the value to 15 minutes or 4 hours depending on your use case. (The default should be 15 seconds)

### Step 3: Run teleop

Start multiple terminals to run the following programs.

- Activate Franka, and launch the Polymetis server:
```bash
python launch_robot.py robot_client=franka_hardware use_real_time=true
```

- Launch the gripper server:
```bash
python launch_gripper.py
```

- Run teleop script:
```bash
cd fairo/polymetis
python examples/4_teleoperation.py
```
