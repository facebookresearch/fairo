"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import atexit
import logging
import numpy as np
import json
import pickle
import os
import shutil
import subprocess
import tempfile
import time
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils import (
    place_blocks,
    edit_cuberite_config,
)
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.wait_for_cuberite import (
    wait_for_cuberite,
)
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import (
    build_shape_scene,
    build_extra_simple_shape_scene,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)

repo_home = os.path.dirname(os.path.realpath(__file__))

DEFAULT_PORT = 25565


def add_plugins(workdir, plugins):
    """Add plugins to craftassist_cuberite_utils's config"""
    for p in plugins:
        edit_cuberite_config.add_plugin(workdir + "/settings.ini", p)


def create_workdir(
    config_name, seed, game_mode, port, plugins, place_blocks_yzx, workdir_root=None
):
    if workdir_root is None:
        workdir_root = tempfile.mkdtemp()
    os.makedirs(workdir_root, exist_ok=True)
    workdir = os.path.join(workdir_root, "cuberite")
    config_dir = os.path.join(repo_home, "server/cuberite_config", config_name)
    plugins_dir = os.path.join(repo_home, "server/cuberite_plugins")
    shutil.copytree(config_dir, workdir)
    shutil.copytree(plugins_dir, workdir + "/Plugins")
    edit_cuberite_config.set_port(workdir + "/settings.ini", port)
    edit_cuberite_config.set_seed(workdir + "/world/world.ini", seed)
    if game_mode == "survival":
        edit_cuberite_config.set_mode_survival(workdir + "/world/world.ini")
    elif game_mode == "creative":
        edit_cuberite_config.set_mode_creative(workdir + "/world/world.ini")
    else:
        raise ValueError("create_workdir got bad game_mode={}".format(game_mode))
    if place_blocks_yzx is not None:
        generate_place_blocks_plugin(workdir, place_blocks_yzx)
    add_plugins(workdir, plugins)
    return workdir


def generate_place_blocks_plugin(workdir, place_blocks_yzx):
    # Generate place_blocks.lua plugin
    plugin_dir = workdir + "/Plugins/place_blocks"
    plugin_template = plugin_dir + "/place_blocks.lua.template"
    # Read plugin template
    with open(plugin_template, "r") as f:
        template = f.read()
    # Generate lua code
    if type(place_blocks_yzx) == list:
        dicts = place_blocks_yzx
    else:
        dicts = place_blocks.yzx_to_dicts(place_blocks_yzx)
    blocks_to_place = place_blocks.dicts_to_lua(dicts)
    out = template.replace("__BLOCKS_TO_PLACE__", blocks_to_place)
    # Write lua code
    with open(plugin_dir + "/place_blocks.lua", "w") as f:
        f.write(out)
    # Add place_blocks lua plugin to config
    edit_cuberite_config.add_plugin(workdir + "/settings.ini", "place_blocks")


class CuberiteProcess:
    def __init__(
        self,
        config_name,
        seed,
        game_mode="survival",
        port=DEFAULT_PORT,
        plugins=[],
        log_comm_in=False,
        place_blocks_yzx=None,
        workdir_root=None,
    ):
        self.workdir = create_workdir(
            config_name, seed, game_mode, port, plugins, place_blocks_yzx, workdir_root
        )
        logging.info("Cuberite workdir: {}".format(self.workdir))
        popen = [repo_home + "/server/cuberite/Server/Cuberite"]
        if log_comm_in:
            popen.append("--log-comm-in")
        self.p = subprocess.Popen(popen, cwd=self.workdir, stdin=subprocess.PIPE)
        self.cleaned_up = False
        atexit.register(self.atexit)
        wait_for_cuberite("localhost", port)
        logging.info("Cuberite listening at port {}".format(port))

    def destroy(self):
        # Cuberite often segfaults if we kill it while an player is disconnecting.
        # Rather than fix a bug in Cuberite, let's just give Cuberite a moment to
        # gather its bearings.
        time.sleep(0.25)
        self.p.terminate()
        self.p.wait()
        self.cleaned_up = True

    def wait(self):
        self.p.wait()

    def atexit(self):
        if not self.cleaned_up:
            logging.info("Killing pid {}".format(self.p.pid))
            self.p.kill()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flat_world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--log-comm-in", action="store_true")
    parser.add_argument("--mode", choices=["creative", "survival"], default="creative")
    parser.add_argument("--workdir")
    parser.add_argument("--add-plugin", action="append", default=[])
    # a path to a schematic data file.  if the path ends in .npy,
    # it is assumed the schematic is a 5 dim (y,z,x,bid,meta) numpy array
    # if the path ends in .pk, it is assumed the schematic is stored in a
    # pickled list of {"x":x, "y":y", "z":z, "id": bid, "meta": meta} dicts
    # if the path ends in .json, it is assumed the schematic is stored in a
    # json with a list of of {"x":x, "y":y", "z":z, "id": bid, "meta": meta} dicts
    # in a "schematic_for_cuberite" field
    parser.add_argument("--schematic", default="")
    # args for random shape generation, if used:
    parser.add_argument(
        "--mob_config", type=str, default=""
    )  # this doesn't work in cuberite yet!, FIXME
    parser.add_argument("--random_shapes", action="store_true")
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--GROUND_DEPTH", type=int, default=5)
    parser.add_argument("--MAX_NUM_GROUND_HOLES", type=int, default=0)
    parser.add_argument("--extra_simple", action="store_true", default=False)
    parser.add_argument("--iglu_scenes", default="")
    parser.add_argument("--SL", type=int, default=17)
    parser.add_argument("--H", type=int, default=13)
    args = parser.parse_args()

    plugins = ["debug", "chatlog", "point_blocks"] + args.add_plugin
    if args.logging:
        plugins += ["logging"]

    schematic = None
    # if args.schematic, load the schematic when starting
    if args.schematic:
        if args.schematic.endswith(".npy"):
            schematic = np.load(args.schematic)
        elif args.schematic.endswith(".pk"):
            with open(args.schematic, "rb") as f:
                schematic = pickle.load(f)
        else:
            with open(args.schematic, "rb") as f:
                J = json.load(f)
                if type(J) is list and len(J) == 1:
                    J = J[0]
                schematic = J["schematic_for_cuberite"]
        if args.random_shapes:
            # TODO allow both?
            print("warning: ignoring the schematic and using random shapes")
    if args.random_shapes:
        args.cuberite_y_offset = 63 - args.GROUND_DEPTH
        args.cuberite_x_offset = -args.SL // 2
        args.cuberite_z_offset = -args.SL // 2
        if args.extra_simple:
            schematic = build_extra_simple_shape_scene(args)["schematic_for_cuberite"]
        else:
            schematic = build_shape_scene(args)["schematic_for_cuberite"]

    p = CuberiteProcess(
        args.config,
        seed=args.seed,
        game_mode=args.mode,
        port=args.port,
        plugins=plugins,
        log_comm_in=args.log_comm_in,
        place_blocks_yzx=schematic,
        workdir_root=args.workdir,
    )
    p.wait()
