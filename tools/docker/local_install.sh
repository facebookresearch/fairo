#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# clean up dangling images to get back disk space for use
docker rmi -f $(docker images -f "dangling=true" -q)

docker build -t minecraft -f ./tools/docker/Dockerfile.client .


if [ "$(docker ps -aq -f name="minecraft_container")" ]; then
    # clean up old container
    docker stop "minecraft_container"
    docker rm "minecraft_container"
fi

# build new container
docker run -it -p 25565:25565 -p 3000:3000 -p 2556:2556 -p 2557:2557 -p 9000:9000 -p 5000:5000  -v "$(pwd)"/python:/minecraft/python -v "$(pwd)"/annotation_tools:/minecraft/annotation_tools --name minecraft_container minecraft
