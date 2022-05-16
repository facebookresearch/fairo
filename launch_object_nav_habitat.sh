#!/usr/bin/env bash
set -ex

pushd droidlet/lowlevel/locobot/remote
LOCOBOT_IP=0.0.0.0 ./launch_pyro_habitat.sh --scene_path Replica-Dataset/apartment_0/habitat/mesh_semantic.ply
popd

pushd agents/locobot
python teleop.py
popd

# pushd droidlet/lowlevel/locobot/remote
# REPLICA_SCENES=`find Replica-Dataset -name "*mesh_semantic.ply"`
# MP3D_SCENES=`find Matterport-Dataset -name "*.glb"`
# SCENES="$REPLICA_SCENES[@] $MP3D_SCENES[@]"
# popd

# for scene in $SCENES
# do
#     pushd droidlet/lowlevel/locobot/remote
#     echo "Starting object nav task in scene $scene"
#     LOCOBOT_IP=0.0.0.0 ./launch_pyro_habitat.sh --scene_path $scene
#     popd

#     pushd agents/locobot
#     python teleop.py
#     popd
# done