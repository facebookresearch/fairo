"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import glob
import logging
import numpy as np
import os
import pathlib
import subprocess
import sys

python_dir = pathlib.Path(__file__).parents[0].absolute()
sys.path.insert(0, python_dir)

from cuberite_process import CuberiteProcess
from repo import repo_home

logging.basicConfig(stream=sys.stdout, format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


def to_unit_vec(yaw, pitch):
    pitch *= 3.14159 / 180
    yaw *= 3.14159 / 180
    return np.array(
        [-1 * np.cos(pitch) * np.sin(yaw), -1 * np.sin(pitch), np.cos(pitch) * np.cos(yaw)]
    )


def ground_height(blocks):
    """Return ground height"""
    dirt_pct = np.mean(np.mean(blocks[:, :, :, 0] == 2, axis=1), axis=1)
    if (dirt_pct > 0.25).any():
        return np.argmax(dirt_pct)
    return None


def render(
    npy_file, out_dir, world, seed, no_chunky, no_vision, port, distance, yaws, spp, img_size
):
    """This function renders the npy_file in the world using cuberite"""

    # step 1: create the world with Cuberite, with npy blocks placed
    logging.info("Launching cuberite at port {}".format(port))
    if npy_file != "":
        schematic = np.load(npy_file)
        p = CuberiteProcess(
            world, seed=0, game_mode="creative", place_blocks_yzx=schematic, port=port
        )
    else:
        schematic = None
        p = CuberiteProcess(
            world, seed=0, game_mode="creative", port=port, plugins=["debug", "spawn_position"]
        )
    logging.info("Destroying cuberite at port {}".format(port))
    p.destroy()

    world_dir = os.path.join(p.workdir, "world")

    print("==========================")
    print("WORLD directory:", world_dir)
    print("==========================")

    region_dir = os.path.join(world_dir, "region")
    mca_files = glob.glob(os.path.join(region_dir, "*.mca"))
    assert len(mca_files) > 0, "No region files at {}".format(region_dir)

    # step 2: render view binary
    render_view_bin = os.path.join(repo_home, "bin/render_view")
    assert os.path.isfile(
        render_view_bin
    ), "{} not found.\n\nTry running: make render_view".format(render_view_bin)

    if schematic is not None:
        # render a numpy object, set focus and distance
        # remove blocks below ground-level
        g = ground_height(schematic)
        schematic = schematic[(g or 0) :, :, :, :]

        ymax, zmax, xmax, _ = schematic.shape
        ymid, zmid, xmid = ymax // 2, zmax // 2, xmax // 2
        focus = np.array([xmid, ymid + 63, zmid])  # TODO: +63 only works for flat_world seed=0

        if distance is None:
            distance = int((xmax ** 2 + zmax ** 2) ** 0.5)
    else:
        f = open(p.workdir + "/spawn.txt", "r")
        playerx, playery, playerz = [float(i) for i in f.read().split("\n")]
        f.close()

        print("Spawn position:", playerx, playery, playerz)

    procs = []
    if yaws is None:
        yaws = range(0, 360, 90)
    for yaw in yaws:
        look = [yaw, 0]
        look_xyz = to_unit_vec(*look)

        if schematic is not None:
            camera = focus - (look_xyz * distance)
        else:
            camera = np.array([playerx, playery + 1.6, playerz])

        if not no_vision:
            call = [
                render_view_bin,
                "--out-prefix",
                out_dir + "/vision.%d" % yaw,
                "--mca-files",
                *mca_files,
                "--camera",
                *camera,
                "--sizes",
                *img_size,
                "--look",
                yaw,
                0,
                "--block",
                1,
                "--depth",
                1,
                "--blockpos",
                1,
            ]
            call = list(map(str, call))

            logging.info("CALL: " + " ".join(call))
            procs.append(subprocess.Popen(call))

        if not no_chunky:
            call = [
                "python",
                "{}/minecraft_render/render.py".format(repo_home),
                "--world",
                world_dir,
                "--out",
                "{}/chunky.{}.png".format(out_dir, yaw),
                "--camera",
                *camera,
                "--look",
                *look,
                "--size",
                *img_size,
                "--spp",
                spp,
                "--chunk-min",
                -10,
                "--chunk-max",
                10,
            ]
            call = list(map(str, call))
            logging.info("CALL: " + " ".join(call))
            procs.append(subprocess.Popen(call))

    for p in procs:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy_schematic",
        default="",
        help="a 3D numpy array (e.g., a house) to render in the world, if missing only the world ",
    )
    parser.add_argument(
        "--out-dir", "-o", required=True, help="Directory in which to write vision files"
    )
    parser.add_argument(
        "--world", default="flat_world", help="world style, [flat_world, diverse_world]"
    )
    parser.add_argument("--seed", default=0, type=int, help="Value of seed")
    parser.add_argument(
        "--no-chunky", action="store_true", help="Skip generation of chunky (human-view) images"
    )
    parser.add_argument("--no-vision", action="store_true", help="Skip generation of agent vision")
    parser.add_argument("--distance", type=int, help="Distance from camera to schematic center")
    parser.add_argument("--yaws", type=int, nargs="+", help="Angles from which to take photos")
    parser.add_argument("--spp", type=int, default=100, help="samples per pixel")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256])
    args = parser.parse_args()

    render(
        args.npy_schematic,
        args.out_dir,
        args.world,
        args.seed,
        args.no_chunky,
        args.no_vision,
        args.port,
        args.distance,
        args.yaws,
        args.spp,
        args.size,
    )
