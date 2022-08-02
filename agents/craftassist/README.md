The goal of this project is to build an intelligent, collaborative assistant in the game of [Minecraft](https://www.minecraft.net/en-us/)<sup>1</sup> that can perform a wide variety of tasks specified by human players.

A detailed outline and documentation is available in [this paper](https://arxiv.org/abs/1907.08584)

This release is motivated by a long-term research agenda described [here](https://research.fb.com/publications/why-build-an-assistant-in-minecraft/).

![GIF of Gameplay With Bot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/bot_46.gif)

# Installation & Getting Started

The recommended way to install CraftAssist is by building a [Docker](https://docker.com) container. This approach enables you to run the setup on any machine and dependencies are taken care of. Alternatively, you can install it on your local machine by setting up the environment and installing all required packages manually as explained [here](#option-b-local-installation)

## Option A: Docker Installation (Recommended)

### Prerequisite

First [install Docker](https://docs.docker.com/get-docker/) in your local system. Then go to your Docker resource settings and make sure that the memory allocated to Docker is at least 4GB (smaller amounts of memory can result in the build crashing).

### Build docker container

First run:
```
./tools/docker/local_install.sh
```
This step can take some time as the docker container is being built.
If the above fails at any step because of issues with submodules :
- If you haven't initialized all submodules, do :
```
git submodule update --init --recursive
```

- If you need to update all submodules, use:
```
git submodule update --recursive
```

Once the install step completes, the docker container has now been created successfully and you'll be automatically logged into the container.
If you aren't already in the container shell (because you didn't do the step above), run:
```
docker start droidlet_container
```

to start the docker container. You have now successfully started the docker container, once you've done this you can directly go to [start the environment](#start-the-environment) and [starting the agent process](#start-the-agent-process) steps from here.

For the remaining sections of this tutorial, if you need to run commands in a new shell window of this built container, simply run:
```
docker exec -it droidlet_container bash
```
to start a new shell window.

After you are done playing with the bot, run:
```
docker stop droidlet_container
```
to stop the docker container.


## Option B: Local Installation

### Dependencies

Make sure the following packages have already been installed before moving on:
* CMake
* Python3
* Glog
* Boost
* Eigen
* For Mac users:
  * LLVM version < 10 to successfully use clang. [Working with multiple versions of Xcode](https://medium.com/@hacknicity/working-with-multiple-versions-of-xcode-e331c01aa6bc).
  

### Building client and server

To build Cuberite and the C++ Minecraft client:
```
cd droidlet/lowlevel/minecraft
make
```

## Start the environment

Currently we support two types of environment (backend): the first one is cuberite and the second one is pyworld. We recommend you to use pyworld backend.

To start the environment:

### Option A: Pyworld (Recommended)

Run the following command:

```
python droidlet/lowlevel/minecraft/pyworld/run_world.py
```
to start a pyworld backend listening on `localhost:6002`

### Option B: Cuberite

Run the following command:

```
python droidlet/lowlevel/minecraft/cuberite_process.py --config flat_world
```
to start a cuberite instance listening on `localhost:25565`

## Start the agent process

Once you have the environment up and running, simply run the following command to start the agent process:

```
python agents/craftassist/craftassist_agent.py --backend pyworld --port 6002
```

Then you will be able to open a web dashboard on `localhost:8000` where you can inspect the agent internal state and interact with our agent.

## Connecting your Minecraft game client

Optionally you can interact with our agent in the real Minecraft game (Keep in mind that you can always interact with our agent via the web dashboard `localhost:8000` without having to buy a minecraft client)

Buy and download the [official Minecraft client](https://my.minecraft.net/en-us/store/minecraft/).
You can inspect the world and view the Minecraft agent's actions by logging into the
running Cuberite instance from the game client.

To connect the client to the running Cuberite instance, click in the Minecraft client:
```
Multiplayer > Direct Connect > localhost:25565
```

#### Error: Unsupported Protocol Version

Our Cuberite system supports at most v1.12 version of Minecraft client.
[Please follow these instructions](https://help.minecraft.net/hc/en-us/articles/360034754852-Changing-game-versions-) to add a 1.12.x profile and use it to connect.

## Running tests

```
./tests/test.sh
```

## Citation

If you would like to cite this repository in your research, please cite [the CraftAssist paper](https://arxiv.org/abs/1907.08584).
```
@misc{gray2019craftassist,
    title={CraftAssist: A Framework for Dialogue-enabled Interactive Agents},
    author={Jonathan Gray and Kavya Srinet and Yacine Jernite and Haonan Yu and Zhuoyuan Chen and Demi Guo and Siddharth Goyal and C. Lawrence Zitnick and Arthur Szlam},
    year={2019},
    eprint={1907.08584},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```