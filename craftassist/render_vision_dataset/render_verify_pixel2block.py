"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import subprocess
import random
import cv2

import numpy as np

import sys

python_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, python_dir)

from cuberite_process import CuberiteProcess
from repo import repo_home

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


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


def change_block(schematic, b):
    x, y, z = b
    ## change to red wool
    schematic[y][z][x][0] = 35
    schematic[y][z][x][1] = 14


def render(npy_p2b, out_dir, port, spp, img_size, mn=None):
    npy_file = (
        os.path.expanduser("~")
        + "/minecraft_houses/"
        + ".".join(npy_p2b.split(".")[1:-2])
        + "/schematic.npy"
    )

    schematic = np.load(npy_file)
    print(schematic.shape)
    house_name = os.path.basename(os.path.dirname(npy_file))
    p2b = np.load(npy_p2b)

    # remove blocks below ground-level
    g = ground_height(schematic)
    schematic = schematic[(g or 0) :, :, :, :]
    ys, zs, xs = np.nonzero(schematic[:, :, :, 0] > 0)

    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    focus = np.array([xmid, ymid + 63, zmid])  # TODO: +63 only works for flat_world seed=0w

    yaw, distance = list(map(int, npy_p2b.split(".")[-2].split("_")))

    look = [yaw, 0]
    look_xyz = to_unit_vec(*look)
    camera = focus - (look_xyz * distance)

    if mn == [0, 0]:
        M, N = p2b.shape[:2]
        while True:
            m = random.randint(0, M - 1)
            n = random.randint(0, N - 1)
            if p2b[m][n][0] != -1:
                break
    else:
        m, n = mn

    print("Select pixel at {}".format((m, n)))
    print("Mapped block {}".format(p2b[m][n]))
    change_block(schematic, p2b[m][n])

    logging.info("Launching cuberite at port {}".format(port))
    p = CuberiteProcess(
        "flat_world", seed=0, game_mode="creative", place_blocks_yzx=schematic, port=port
    )
    logging.info("Destroying cuberite at port {}".format(port))
    p.destroy()

    world_dir = os.path.join(p.workdir, "world")

    render_view_bin = os.path.join(repo_home, "bin/render_view")
    assert os.path.isfile(
        render_view_bin
    ), "{} not found.\n\nTry running: make render_view".format(render_view_bin)

    procs = []

    chunky_id = "{}_{}".format(yaw, distance)
    out_file = "{}/chunky_verify.{}.{}.png".format(out_dir, house_name, chunky_id)

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

    ## draw the sampled pixel for a better view
    img = cv2.imread(out_file)
    cv2.circle(img, (n, m), 2, (255, 0, 0))
    cv2.imwrite(out_file, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_p2b")
    parser.add_argument(
        "--out-dir", "-o", required=True, help="Directory in which to write vision files"
    )
    parser.add_argument("--spp", type=int, default=25, help="samples per pixel")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--size", type=int, nargs=2, default=[300, 225])
    parser.add_argument("--mn", type=int, nargs=2, default=[0, 0])
    args = parser.parse_args()

    render(args.npy_p2b, args.out_dir, args.port, args.spp, args.size, args.mn)
