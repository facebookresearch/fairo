This folder introduces the Locobot Assistant, which is a droidlet agent embodied on a [Locobot](http://www.locobot.org/). We also support a virtual backend for the Locobot Assistant in the simulation platform [Habitat](https://aihabitat.org/).

<center>

<p align="center">
  <table align="center">
    <tr>
      <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/loco_physical.gif" width=170 height=270></td>
      <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/habitat_mover.gif" width=400 height=270></td>
    </tr>
    <tr>
      <td>Locobot Assistant in the real world.</td>
      <td>Locobot Assistant being teleoperated in Habitat.</td>
    </tr>
  </table>
</p>

## Getting Started 

<p align="center">
  <table align="center">
    <thead><th>Physical Locobot</th>
        <th>Habitat</th>
    </thead>
    <tr valign="top">        
        <td> 1. <a href="https://github.com/facebookresearch/pyrobot/blob/master/README.md#installation"> Setup PyRobot<a> using Python 3 on the Locobot.
        <sub><pre lang="bash">
./locobot_install_all.sh -t full -p 3 -l interbotix
        </pre></sub></td>
        <td>We provide a docker image for habitat that comes bundled with PyRobot.
        <sub><pre lang="bash">
docker pull theh1ghwayman/locobot-assistant:segm
        </pre></sub></td>
    </tr>
    <tr valign="top">        
      <td> 2. Launch Pyro4 <p> Copy the <a href="https://github.com/facebookresearch/droidlet/tree/main/locobot/robot"> robot<a/> folder onto the Locobot and then do the following: </p>
        <sub><pre lang="bash">
cd robot
chmod +x launch_pyro.sh
./launch_pyro.sh
        </pre></sub></td>
        <td><sub><pre lang="bash">
        <br/>
docker run --gpus all -it --rm --ipc=host -v $(pwd):/remote -w /remote theh1ghwayman/locobot-assistant:segm bash
roscore &
load_pyrobot_env
cd droidlet/lowlevel/locobot/remote
./launch_pyro_habitat.sh
        </pre></sub></td>
    </tr>
        <tr valign="top">
        <td colspan=3> 3. Run the locobot assistant.
        <sub><pre lang="bash">
export LOCOBOT_IP="IP of the locobot"
python locobot_agent.py
        </pre></sub>
        </td>      
    </tr>
    <tr valign="top">
        <td colspan=3> 4. Open the dashboard to interact with the assistant on `localhost:8000` 
          (only currently supported on Google Chrome).
        </td>      
    </tr>    
  </table>
</p>



The Locobot Assistant has some default behaviors (like SLAM mapping shown below), which are things it does when it is idle.

<p align="center">
    <img src="https://media.giphy.com/media/XwmXCvoGHBXBqYUdMe/giphy.gif", width="600" height="192">
</p>


To run the Locobot Assistant without these default behaviors, use the `--no_default_behavior` flag.
```
python locobot_agent.py --no_default_behavior
```
