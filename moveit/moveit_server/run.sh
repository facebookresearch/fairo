#!/bin/bash
cd "$(dirname "$0")"

set -e

docker build -t fbrp/moveit -f moveit.Dockerfile .

mkdir -p /tmp/mesh

docker run \
  --rm \
  -it \
  --ipc=host \
  --pid=host \
  -v /tmp/mesh:/tmp/mesh \
  --name panda_planner \
  fbrp/moveit
