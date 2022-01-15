# Teleoperation Setup Guide

## Step 1: Install Polymetis at the dev branch locally
```bash
git clone --recurse-submodules git@github.com:facebookresearch/fairo
cd fairo
git checkout austinw/oculus_teleop
```
Installation instructions can be found [here](https://polymetis-docs.github.io/installation.html#from-source).

This will also create a new conda env for everything else to be installed into.

## Step 2: Install additional pip dependencies needed for the experimental teleop code
```bash
pip install sophuspy getch
```

## Step 3: Setup Oculus

Install oculus-reader:
```bash
git clone git@github.com:rail-berkeley/oculus_reader
cd oculus_reader
pip install .
```

Then, follow instructions in the [README](https://github.com/rail-berkeley/oculus_reader/blob/main/README.md) to set up with a Oculus Quest headset.

Note: If `adb devices` does not list the Oculus device or has the error `no permissions (user in plugdev group; are your udev rules wrong?);`, restart adb with sudo access:
```bash
adb kill-server
sudo adb start-server
```

To prevent the headset from going to sleep, go to Settings --> Device --> Auto Sleep Headset and change the value to 15 minutes or 4 hours depending on your use case. (The default should be 15 seconds)

## Step 4: Setup workspace

The current teleoperation script treats the left controller as a static frame of reference and uses the right controller to operate the arm.

To setup the static frame, put on the Oculus Quest headset. 
If previous steps are done correctly, you should be able to see a primitive 3D scene with axes indicators on both controllers. 
Place the left controller in a static position so that the axes of the controller (RGB corresponding to XYZ) are parallel with that of the Franka arm coordinate system.

Then, take off the headset and place it in a location less than 2 meters away where it has a good view of the workspace.

## Step 5: Run teleop

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

Controls:
- Using the right controller, fully press the grip button (middle finger) to engage teleoperation. Release to stop at any time.
- Hold B to perform grasp.
- For safety, it is recommended to hold the emergency stop button in your other hand while controlling the robot through teleoperation.

### Keyboard debugging

Keyboard control mode can be activated by running the script as follows:
```bash
python examples/4_teleoperation.py --keyboard
```

Controls:
- W: Move 1cm in +x direction
- A: Move 1cm in -x direction
- S: Move 1cm in +y direction
- D: Move 1cm in -y direction
- R: Move 1cm in +z direction
- F: Move 1cm in -z direction
- space: Toggle gripper state (open/close)
