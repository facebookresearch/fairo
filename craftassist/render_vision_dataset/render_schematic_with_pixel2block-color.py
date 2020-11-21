"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import subprocess
import random
from sklearn.neighbors import KDTree
import cv2
import pickle
import numpy as np

import sys

python_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, python_dir)

from cuberite_process import CuberiteProcess
from repo import repo_home

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v


with open(os.path.expanduser("~") + "/minecraft/minecraft_specs/block_images/rgbs.pkl", "rb") as f:
    block_colors = pickle.load(f)
    wool_blocks = [(35, 1), (35, 2), (35, 4), (35, 5), (35, 11)]
    metaid_to_hue = {b[1]: rgb2hsv(*(block_colors[b]))[0] for b in wool_blocks}


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


def randomly_change_blocks(schematic):
    new_schematic = np.copy(schematic)
    ymax, zmax, xmax, _ = new_schematic.shape
    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if new_schematic[y][z][x][0] > 0:
                    ## change all non-air blocks to a random wool block
                    new_schematic[y][z][x][0] = 35
                    new_schematic[y][z][x][1] = random.choice(list(metaid_to_hue.keys()))

    return new_schematic


def add_new_schematic_hue(schematic_hue, new_schematic, i):
    ymax, zmax, xmax = new_schematic.shape
    new_schematic_hue = np.zeros((ymax, zmax, xmax), dtype=np.int32)
    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if new_schematic[y][z][x] in metaid_to_hue:
                    new_schematic_hue[y][z][x] = metaid_to_hue[new_schematic[y][z][x]]
                else:
                    new_schematic_hue[y][z][x] = random.randint(-20000, -10000)

    schematic_hue[:, :, :, i] = new_schematic_hue


def render(npy_file, out_dir, port, spp, img_size):
    if "p2b" in npy_file:  ## we're going to re-compute the correspondence
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
    ys, zs, xs = np.nonzero(schematic[:, :, :, 0] > 0)

    if len(ys) < 5:
        print("too few non-air blocks; will not render")
        return

    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    focus = np.array([xmid, ymid + 63, zmid])  # TODO: +63 only works for flat_world seed=0w

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

    repeat = 10
    schematic_hue = np.zeros(schematic.shape[:3] + (repeat - 1,), dtype=np.int32)
    tmp_images = []
    for i in range(repeat):
        if i < repeat - 1:
            new_schematic = randomly_change_blocks(schematic)
            add_new_schematic_hue(schematic_hue, new_schematic[:, :, :, 1], i)
        else:
            break  # do not render the full image again
            new_schematic = schematic
            img_size = [s * 3 for s in img_size]

        logging.info("Launching cuberite at port {}".format(port))
        p = CuberiteProcess(
            "flat_world", seed=0, game_mode="creative", place_blocks_yzx=new_schematic, port=port
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
        out_file = "{}/chunky.{}.{}.png".format(out_dir, house_name, chunky_id)
        if i < repeat - 1:  # tmp out file
            out_file += ".tmp.png"

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

        if i < repeat - 1:
            tmp_images.append(cv2.imread(out_file))
            os.system("rm -f " + out_file)  ## delete the tmp image

    ## now we need to compute the pixel-to-block correspondence
    p2b = pixel2block(tmp_images, schematic_hue)

    ## write the correspondence to disk
    ## x-y-z is after applying ground_height
    p2b_file = "{}/p2b.{}.{}.npy".format(out_dir, house_name, chunky_id)
    np.save(p2b_file, p2b)


def pixel2block(random_images, schematic_hue):
    """
    This function returns a numpy array (M,N,3) that indicates which pixel corresponds to
    which block.

    If a pixel has [-1, -1, -1], then it means this pixel does not map to any block
    """
    for i in range(len(random_images)):
        random_images[i] = cv2.cvtColor(random_images[i], cv2.COLOR_BGR2HSV)

    ## init the ret to all -1s
    ret = np.ones(random_images[0].shape[:2] + (3,), dtype=np.int32) * -1

    ymax, zmax, xmax, _ = schematic_hue.shape

    schematic_hue = np.reshape(schematic_hue, (-1, schematic_hue.shape[-1]))
    kdt = KDTree(schematic_hue, leaf_size=2)

    hue_vecs = []
    for m in range(ret.shape[0]):
        for n in range(ret.shape[1]):
            ## the original range is [0,179]
            hue_vecs.append([img[m, n][0] * 2 for img in random_images])

    hue_vecs = np.reshape(np.array(hue_vecs), (-1, len(random_images)))

    query = kdt.query(hue_vecs, k=1, return_distance=False)

    assert len(query) == ret.shape[0] * ret.shape[1]

    for i in range(len(query)):
        m = i // ret.shape[1]
        n = i % ret.shape[1]
        y = query[i][0] // (zmax * xmax)
        z = (query[i][0] % (zmax * xmax)) // xmax
        x = (query[i][0] % (zmax * xmax)) % xmax
        ret[m][n] = [x, y, z]

    return ret


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
