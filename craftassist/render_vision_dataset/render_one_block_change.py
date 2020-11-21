"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import glob
import logging
import os
import subprocess
import random

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


def ray_intersect_triangle(p0, p1, triangle):
    ## Code taken from:
    #    https://www.erikrotteveel.com/python/three-dimensional-ray-tracing-in-python/
    #
    # Tests if a ray starting at point p0, in the direction
    # p1 - p0, will intersect with the triangle.
    #
    # arguments:
    # p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    # triangle: numpy.ndarray, shaped (3,3), with each row
    #     representing a vertex and three columns for x, y, z.
    #
    # returns:
    #    0 if ray does not intersect triangle,
    #    1 if it will intersect the triangle,
    #    2 if starting point lies in the triangle.
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)

    # Here is the main difference with the code in the link.
    # Instead of returning if the ray is in the plane of the
    # triangle, we set rI, the parameter at which the ray
    # intersects the plane of the triangle, to zero so that
    # we can later check if the starting point of the ray
    # lies on the triangle. This is important for checking
    # if a point is inside a polygon or not.

    if b == 0.0:
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b
    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - np.inner(v, v) * np.inner(w, u)) / denom

    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - np.inner(u, u) * np.inner(w, v)) / denom

    if (ti < 0.0) | (si + ti > 1.0):
        return 0
    if rI == 0.0:
        # point 0 lies ON the triangle. If checking for
        # point inside polygon, return 2 so that the loop
        # over triangles can stop, because it is on the
        # polygon, thus inside.
        return 2
    return 1


def intersect_cube(xyz, camera, focus):
    """
    Test if ray 'focus - camera' intersects with the cube
    '(x, y, z) - (x + 1, y + 1, z + 1)'
    To do this, we check if at least one triangle intersects with
    the ray
    """
    x, y, z = xyz
    triangles = [
        [[x, y, z], [x + 1, y, z], [x + 1, y + 1, z]],
        [[x, y, z], [x, y + 1, z], [x + 1, y + 1, z]],
        [[x, y, z + 1], [x + 1, y, z + 1], [x + 1, y + 1, z + 1]],
        [[x, y, z + 1], [x, y + 1, z + 1], [x + 1, y + 1, z + 1]],
        [[x, y, z], [x + 1, y, z], [x + 1, y, z + 1]],
        [[x, y, z], [x, y, z + 1], [x + 1, y, z + 1]],
        [[x, y + 1, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1]],
        [[x, y + 1, z], [x, y + 1, z + 1], [x + 1, y + 1, z + 1]],
        [[x, y, z], [x, y + 1, z], [x, y + 1, z + 1]],
        [[x, y, z], [x, y, z + 1], [x, y + 1, z + 1]],
        [[x + 1, y, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1]],
        [[x + 1, y, z], [x + 1, y, z + 1], [x + 1, y + 1, z + 1]],
    ]
    for t in triangles:
        if (
            ray_intersect_triangle(
                np.array(camera).astype("float32"),
                np.array(focus).astype("float32"),
                np.array(t).astype("float32"),
            )
            == 1
        ):
            return True
    return False


def change_one_block(schematic, yaw):
    ## remove 'air' blocks whose ids are 0s
    ymax, zmax, xmax, _ = schematic.shape
    ys, zs, xs, _ = np.nonzero(schematic[:, :, :, :1] > 0)
    xyzs = list(zip(*[xs, ys, zs]))

    print("xmax={} ymax={} zmax={}".format(xmax, ymax, zmax))

    max_dist = int((xmax ** 2 + zmax ** 2) ** 0.5 / 2)
    distance_range = (5, max(5, max_dist) + 1)
    min_camera_height = 1  ## the camera shouldn't be underground
    pitch_range = (-60, 10)

    if not xyzs:
        print("all blocks are air!")
        return None, None, None

    while True:
        focus = random.choice(xyzs)  ## randomly select a block as the focus
        pitch = random.randint(*pitch_range)
        distance = random.randint(*distance_range)
        look_xyz = to_unit_vec(*[yaw, pitch])
        camera = focus - (look_xyz * distance)
        if camera[1] <= min_camera_height:
            continue
        intersected = [
            (np.linalg.norm(np.array(xyz) - camera), xyz)
            for xyz in xyzs
            if intersect_cube(xyz, camera, focus)
        ]
        ## sort the blocks according to their distances to the camera
        ## pick the nearest block that intersects with the ray starting at 'camera'
        ## and looking at 'focus'
        intersected = sorted(intersected, key=lambda p: p[0])
        if len(intersected) > 0 and intersected[0][0] >= distance_range[0]:
            ## the nearest block should have a distance > 10
            break

    x, y, z = intersected[0][1]
    ## change a non-zero block to red wool (id: 35, meta: 14)
    schematic[y][z][x] = [35, 14]
    return pitch, camera, [x, y, z]


def render(
    npy_file,
    out_dir,
    no_chunky,
    no_vision,
    port,
    yaw,
    pitch,
    camera,
    pos,
    spp,
    img_size,
    block_change,
):

    schematic = np.load(npy_file)
    house_name = os.path.basename(os.path.dirname(npy_file))

    # remove blocks below ground-level
    g = ground_height(schematic)
    schematic = schematic[(g or 0) :, :, :, :]

    if yaw is None:
        yaw = random.randint(0, 360 - 1)

    if block_change:
        pitch, camera, pos = change_one_block(schematic, yaw)
        # TODO: +63 only works for flat_world seed=0
        camera[1] += 63  ## why??

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
    if not no_vision:
        call = [
            str(a)
            for a in [
                render_view_bin,
                "--out-dir",
                out_dir,
                "--mca-files",
                *mca_files,
                "--camera",
                *camera,
                "--look",
                yaw,
                pitch,
            ]
        ]
        logging.info("CALL: " + " ".join(call))
        procs.append(subprocess.Popen(call))

    if not no_chunky:
        chunky_id = "_".join(map(str, list(map(int, camera)) + [pitch, yaw] + pos))
        call = [
            str(a)
            for a in [
                "python3",
                "{}/minecraft_render/render.py".format(repo_home),
                "--world",
                world_dir,
                "--out",
                "{}/chunky.{}.{}.{}.png".format(out_dir, house_name, chunky_id, int(block_change)),
                "--camera",
                *camera,
                "--look",
                yaw,
                pitch,
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

    return yaw, pitch, camera, pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_schematic")
    parser.add_argument(
        "--out-dir", "-o", required=True, help="Directory in which to write vision files"
    )
    parser.add_argument(
        "--no-chunky", action="store_true", help="Skip generation of chunky (human-view) images"
    )
    parser.add_argument("--no-vision", action="store_true", help="Skip generation of agent vision")
    parser.add_argument("--spp", type=int, default=25, help="samples per pixel")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--size", type=int, nargs=2, default=[300, 225])
    args = parser.parse_args()

    yaw, pitch, camera, pos = render(
        args.npy_schematic,
        args.out_dir,
        args.no_chunky,
        args.no_vision,
        args.port,
        None,
        None,
        None,
        None,
        args.spp,
        args.size,
        True,
    )

    #    yaw, pitch, camera, pos = None, None, None, None

    render(
        args.npy_schematic,
        args.out_dir,
        args.no_chunky,
        args.no_vision,
        args.port,
        yaw,
        pitch,
        camera,
        pos,
        args.spp,
        args.size,
        False,
    )
