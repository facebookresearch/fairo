The goal of this project is to build an intelligent, collaborative assistant in the game of [Minecraft](https://www.minecraft.net/en-us/)<sup>1</sup> that can perform a wide variety of tasks specified by human players.

A detailed outline and documentation is available in [this paper](https://arxiv.org/abs/1907.08584)

This release is motivated by a long-term research agenda described [here](https://research.fb.com/publications/why-build-an-assistant-in-minecraft/).

![GIF of Gameplay With Bot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/bot_46.gif)

# Installation & Getting Started

The recommended way to install CraftAssist is by building a [Docker](https://docker.com) container. This approach enables you to run the setup on any machine and dependencies are taken care of. Alternatively, you can install it on your local machine by setting up the environment and installing all required packages manually as explained [here](#option-b-local-installation)

## Option A: Docker Installation

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

to start the docker container. You have now successfully started the docker container, once you've done this you can directly go to [run cuberite instance](#run-the-cuberite-instance) and [starting your agent](#running-the-interactive-v0-agent) steps from here.
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

## Run the Cuberite instance

#### For Docker user

Run

```
docker exec -it droidlet_container bash
```

to open a shell in the docker container, and then run:
```
python droidlet/lowlevel/minecraft/cuberite_process.py --config flat_world
```
to start an instance of cuberite instance listening on `localhost:25565`

#### For local user

Simply run the following command:
```
python droidlet/lowlevel/minecraft/cuberite_process.py --config flat_world
```
to start an instance of cuberite instance listening on `localhost:25565`


## Connecting your Minecraft game client (so you can see what's happening)

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

## Running the interactive V0 agent

Assuming you have set up the [Cuberite server](#run-the-cuberite-instance)
and the [client](#connecting-your-minecraft-game-client-so-you-can-see-whats-happening)

#### For Docker user

Run:
```
docker exec -it droidlet_container bash
```

to open a shell in the docker container, and then run:
```
python craftassist_agent.py
```

#### For local user

Run:
```
python craftassist_agent.py
```

You should see a new bot player join the game.
Chat with the bot by pressing `t` to open the dialogue box, and `Enter` to submit.
Use the `w`, `a`, `s`, and `d` keys to navigate, left and right mouse clicks to destroy and place blocks, and `e` to open your inventory and select blocks to place.

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