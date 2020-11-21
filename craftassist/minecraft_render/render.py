"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid

# LOOK ANGLES
# -----------------
# - chunky definitions of yaw/pitch differ from minecraft's:
#   -> chunky_pitch = minecraft_pitch - 90
#   -> chunk_yaw = 90 - minecraft_yaw
# - chunky uses radians, minecraft uses degrees


parser = argparse.ArgumentParser()
parser.add_argument("--world", required=True, help="path to world files")
parser.add_argument("--out", "-o", required=True, help="path to write image")
parser.add_argument("--camera", type=float, nargs=3, default=[16, 70, 48])
parser.add_argument("--look", type=float, nargs=2, default=[-90, -90])
parser.add_argument("--focal-offset", type=float, default=30)
parser.add_argument("--chunk-min", type=int, default=-1)
parser.add_argument("--chunk-max", type=int, default=3)
parser.add_argument("--size", type=int, nargs=2, default=[800, 600])
parser.add_argument("--spp", type=int, default=100, help="samples per pixel")

REPO_DIR = os.path.dirname(__file__)
CHUNKY_DIR = os.path.join(REPO_DIR, "chunky")
SCENES_DIR = os.path.join(REPO_DIR, "chunky/scenes")


def gen_scene_json(args, name):
    with open(os.path.join(REPO_DIR, "world.json"), "r") as f:
        j = json.load(f)
    j["name"] = name
    j["world"]["path"] = args.world
    j["camera"]["position"]["x"] = args.camera[0]
    j["camera"]["position"]["y"] = args.camera[1]
    j["camera"]["position"]["z"] = args.camera[2]
    j["camera"]["orientation"]["yaw"] = (90 - args.look[0]) * 3.14159 / 180
    j["camera"]["orientation"]["pitch"] = (args.look[1] - 90) * 3.14159 / 180
    j["camera"]["focalOffset"] = args.focal_offset
    j["chunkList"] = [
        [a, b]
        for a in range(args.chunk_min, args.chunk_max + 1)
        for b in range(args.chunk_min, args.chunk_max + 1)
    ]
    j["width"] = args.size[0]
    j["height"] = args.size[1]
    return json.dumps(j)


if __name__ == "__main__":
    args = parser.parse_args()

    name = str(uuid.uuid4())
    base_call = "java -jar -Dchunk.home={0} {0}/ChunkyLauncher.jar".format(CHUNKY_DIR)

    # Create scene
    scene_json = gen_scene_json(args, name)
    os.makedirs(SCENES_DIR, exist_ok=True)
    scene_json_path = os.path.join(SCENES_DIR, name + ".json")
    with open(scene_json_path, "w") as f:
        f.write(scene_json)
    print("Wrote scene to", scene_json_path)

    # Download minecraft if necessary
    if not os.path.isfile(os.path.join(CHUNKY_DIR, "resources/minecraft.jar")):
        call = "{} -download-mc 1.12".format(base_call)
        subprocess.check_call(call.split(" "))

    # Run chunky
    call = "{} -render {}".format(base_call, name)
    subprocess.check_call(call.split(" "))

    # Move output image
    pngs = glob.glob(os.path.join(SCENES_DIR, "{}*.png".format(name)))
    assert len(pngs) == 1, pngs
    shutil.move(pngs[0], args.out)
    print("Wrote image to", args.out)

    # Clean up
    for f in glob.glob(os.path.join(SCENES_DIR, "*{}*".format(name))):
        os.remove(f)
