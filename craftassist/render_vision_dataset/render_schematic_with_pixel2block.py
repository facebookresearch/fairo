"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import glob
import logging
import os
import subprocess
import random
import struct
import numpy as np
import sys

python_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, python_dir)

from cuberite_process import CuberiteProcess
from repo import repo_home

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


y_offset = 63


def to_unit_vec(yaw, pitch):
    pitch *= 3.14159 / 180
    yaw *= 3.14159 / 180
    return np.array(
        [-1 * np.cos(pitch) * np.sin(yaw), -1 * np.sin(pitch), np.cos(pitch) * np.cos(yaw)]
    )


def ground_height(blocks):
    dirt_pct = np.mean(np.mean(blocks[:, :, :, 0] == 2, axis=1), axis=1)
    if (dirt_pct > 0.25).any():
        return np.argmax(dirt_pct)
    return None


def render(npy_file, out_dir, port, spp, img_size):
    no_chunky = "p2b" in npy_file

    if no_chunky:  ## we're going to re-compute the correspondence
        npy_file = os.path.basename(npy_file)
        tokens = npy_file.split(".")
        yaw, distance = list(map(int, tokens[-2].split("_")))
        npy_file = (
            os.path.expanduser("~")
            + "/minecraft_houses/"
            + ".".join(tokens[1:-2])
            + "/schematic.npy"
        )
    else:
        yaw, distance = None, None

    schematic = np.load(npy_file)
    house_name = os.path.basename(os.path.dirname(npy_file))

    # remove blocks below ground-level
    g = ground_height(schematic)
    schematic = schematic[(g or 0) :, :, :, :]
    Y, Z, X, _ = schematic.shape
    ys, zs, xs = np.nonzero(schematic[:, :, :, 0] > 0)

    if len(ys) < 5:
        print("too few non-air blocks; will not render")
        return

    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    focus = np.array([xmid, ymid + y_offset, zmid])  # TODO: +63 only works for flat_world seed=0w

    if yaw is None:
        yaw = random.randint(0, 360 - 1)
        sorted_xs = sorted(xs)
        sorted_zs = sorted(zs)
        N = len(xs)
        ## remove head and tail 2%
        X = sorted_xs[-N // 100] - sorted_xs[N // 100]
        Z = sorted_zs[-N // 100] - sorted_zs[N // 100]
        distance = max(X, Z)

    look = [yaw, 0]
    look_xyz = to_unit_vec(*look)
    camera = focus - (look_xyz * distance)

    logging.info("Launching cuberite at port {}".format(port))
    p = CuberiteProcess(
        "flat_world", seed=0, game_mode="creative", place_blocks_yzx=schematic, port=port
    )
    logging.info("Destroying cuberite at port {}".format(port))
    p.destroy()

    world_dir = os.path.join(p.workdir, "world")
    region_dir = os.path.join(world_dir, "region")
    mca_files = glob.glob(os.path.join(region_dir, "*.mca"))
    assert len(mca_files) > 0, "No region files at {}".format(region_dir)

    render_view_bin = os.path.join(repo_home, "bin/render_view")
    assert os.path.isfile(
        render_view_bin
    ), "{} not found.\n\nTry running: make render_view".format(render_view_bin)

    procs = []

    chunky_id = "{}_{}".format(yaw, distance)
    out_file = "{}/chunky.{}.{}.png".format(out_dir, house_name, chunky_id)
    out_bin_prefix = "{}/{}.{}".format(out_dir, house_name, chunky_id)

    call = [
        str(a)
        for a in [
            render_view_bin,
            "--out-prefix",
            out_bin_prefix,
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
            0,
            "--depth",
            0,
            "--blockpos",
            1,
        ]
    ]
    logging.info("CALL: " + " ".join(call))
    procs.append(subprocess.Popen(call))

    if not no_chunky:
        ## when re-computing the
        call = [
            str(a)
            for a in [
                "python3",
                "{}/minecraft_render/render.py".format(repo_home),
                "--world",
                world_dir,
                "--out",
                out_file,
                "--camera",
                *camera,
                "--look",
                yaw,
                0,
                "--size",
                *img_size,
                "--spp",
                spp,
            ]
        ]
        logging.info("CALL: " + " ".join(call))
        procs.append(subprocess.Popen(call))

    for p in procs:
        p.wait()

    ## read the output blockpos bin and convert it to a npy file
    p2b = read_pixel2block(out_bin_prefix + ".blockpos.bin", X, Y, Z, img_size[0], img_size[1])
    os.system("rm -f {}".format(out_bin_prefix + ".blockpos.bin"))

    ## write the correspondence to disk
    ## x-y-z is after applying ground_height
    p2b_file = "{}/p2b.{}.{}.npy".format(out_dir, house_name, chunky_id)
    np.save(p2b_file, p2b)


def read_pixel2block(blockpos_bin, X, Y, Z, width, height):
    with open(blockpos_bin, "rb") as f:
        content = f.read()

    xyz = struct.unpack(width * height * 3 * "i", content)
    xyz = np.array(xyz, dtype=np.int32)
    p2b = xyz.reshape(height, width, 3)

    for h in range(height):
        for w in range(width):
            x, y, z = p2b[h][w]
            y -= y_offset
            ## check if the block is not on the house
            if x < 0 or x >= X or y < 0 or y >= Y or z < 0 or z >= Z:
                p2b[h][w] = [-1, -1, -1]
            else:
                p2b[h][w] = [x, y, z]

    return p2b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy")
    parser.add_argument(
        "--out-dir", "-o", required=True, help="Directory in which to write vision files"
    )
    parser.add_argument("--spp", type=int, default=25, help="samples per pixel")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--size", type=int, nargs=2, default=[300, 225])
    args = parser.parse_args()

    render(args.npy, args.out_dir, args.port, args.spp, args.size)
