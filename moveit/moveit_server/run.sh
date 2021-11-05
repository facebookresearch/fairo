#!/bin/bash
cd "$(dirname "$0")"

set -e

docker build -t fbrp/moveit -f moveit.Dockerfile .

docker run \
  --rm \
  -it \
  --ipc=host \
  --pid=host \
  --name panda_planner \
  fbrp/moveit
