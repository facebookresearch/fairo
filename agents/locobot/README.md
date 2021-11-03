This folder introduces the Robot Assistant, which is a droidlet agent embodied on a physical or virtual robot.

We currently support [Hello Robot Stretch](https://hello-robot.com/).
We also support a virtual backend for the Assistant in the simulation platform [Habitat](https://aihabitat.org/).

<center>

<p align="center">
  <table align="center">
    <tr>
      <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/loco_physical.gif" width=170 height=270></td>
      <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/habitat_mover.gif" width=400 height=270></td>
    </tr>
    <tr>
      <td>the Assistant in the real world. (Operated on a locobot which is no longer supported)</td>
      <td>the Assistant being teleoperated in Habitat.</td>
    </tr>
  </table>
</p>

## Getting Started 

Make sure you have a working `conda` install. We prefer [MiniConda](https://docs.conda.io/en/latest/miniconda.html)

Create the following environment on a server machine with an NVIDIA GPU that PyTorch supports.

```bash
conda install mamba -c conda-forge -y

mamba create -n droidlet python=3.7 \
    --file ../../conda.txt --file /conda.txt \
    -c pytorch -c aihabitat -c open3d-admin -c conda-forge -y

conda activate droidlet

pip install -r /pip.txt
```

### Running the agent with Habitat

```bash
mamba install https://anaconda.org/aihabitat/habitat-sim/0.2.1/download/linux-64/habitat-sim-0.2.1-py3.7_headless_linux_fc7fb11ccec407753a73ab810d1dbb5f57d0f9b9.tar.bz2
```

In a first shell run:

```bash
conda activate droidlet
cd ../../droidlet/lowlevel/locobot/remote
./launch_pyro_habitat.sh
```

In a second shell, run the agent

```bash
python locobot_agent.py --backend habitat
```

Now, you can access the dashboard to interact with the robot, give commands to the robot and to teleop the robot at http://localhost:8000

### Running the agent with Hello Robot Stretch


1. Clone the `fairo` repository in a shell on the Stretch
2. Run the commands listed at [droidlet/lowlevel/hello_robot/remote](https://github.com/facebookresearch/fairo/blob/main/droidlet/lowlevel/hello_robot/remote/README.md)
3. Note down the IP address of the Hello Robot. When you run the last command in (2), it would say something like `Binding to Host IP 192.168.1.130`, and `192.168.1.130` is the [IP address].

Once the servers have been started on the Hello Robot via (2), then on the current machine where you've created the `conda` environment, run:

```bash
python locobot_agent.py --backend hellorobot --ip [IP address]
```

Now, you can access the dashboard to interact with the robot, give commands to the robot and to teleop the robot at http://localhost:8000

## Other Notes

The Robot Assistant has some default behaviors (like SLAM mapping shown below), which are things it does when it is idle.

<p align="center">
    <img src="https://media.giphy.com/media/XwmXCvoGHBXBqYUdMe/giphy.gif", width="600" height="192">
</p>


To run the Assistant without these default behaviors, use the `--no_default_behavior` flag.
```
python locobot_agent.py --no_default_behavior
```
