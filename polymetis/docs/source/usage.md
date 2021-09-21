# Usage

## Running the server and robot client (simulation)

To run the server and robot client for Franka:
```bash
launch_robot.py robot_client=franka_sim use_real_time=false gui=[true|false]
```

`launch_robot.py` is an executable script which starts both a controller manager server and a robot client which connects to it, and is installed as a script with the conda package by default in `$CONDA_PREFIX/bin`. It utilizes [Hydra](https://hydra.cc/) configurations to easily swap between simulation and hardware.

For simulation, the `gui` option controls whether the PyBullet simulation renders, and requires an active display (i.e. you may need to use VNC if `ssh`ing).

### Running the server and robot client on Franka Panda hardware

1. Wait for your robot to boot up (wait for the blinking yellow light turns solid yellow).
1. If your robot has its joints locked (yellow light), unlock it from Franka Desk.
1. If your robot's external activation device (EAD) is not released (white light), release it (blue light). (If you disconnect from the robot for any reason, you need to re-release the EAD.)

Then:

```bash
# On the NUC
launch_robot.py robot_client=franka_hardware
```

For hardware, `use_real_time` acquires the necessary `sudo` rights for real-time optimizations (see [here](https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/include/real_time.hpp) for the specific optimizations.)
### Running controllers in user code

Once you start an instance of `launch_robot.py`, you can send commands by utilizing the example user code, found in [scripts](https://github.com/facebookresearch/fairo/tree/main/polymetis/examples).

For real-time uses (e.g. on hardware), it is highly recommended that this is run on a different machine than the machine directly controlling the robot, to avoid interruptions of the real-time loop. To do this, simply edit the `ip` while constructing the `RobotInterface`, e.g. in the [scripts](https://github.com/facebookresearch/fairo/tree/main/polymetis/examples).

To **debug custom TorchScript policies**, TorchScript can directly script `print` statements, which will output **on the server** (i.e. for hardware, it will print on the NUC machine, where the `launch_robot.py` script was launched).
