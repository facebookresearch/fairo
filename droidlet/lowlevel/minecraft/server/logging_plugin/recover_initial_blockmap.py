"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import shutil
import subprocess
import tempfile

import sys

python_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, python_dir)

import edit_cuberite_config
from base_log_reader import BaseLogReader

PLUGIN_NAME = "recover_initial"


def recover_initial_blockmap(old_workdir):
    """Given a logdir containing a logging.bin, regenerate the initial blockmap
    and return the directory with the region (.mca) files.
    """

    workdir = tempfile.mkdtemp()
    print("Workdir:", workdir, flush=True)
    repo_home = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")

    # Copy files from old workdir
    paths = ["Plugins", "settings.ini", "blocks.json", "world/world.ini"]
    for p in paths:
        src = os.path.join(old_workdir, p)
        dst = os.path.join(workdir, p)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)

    # Remove logging plugin, add recovery plugin
    settings_ini = os.path.join(workdir, "settings.ini")
    plugins_dir = os.path.join(workdir, "Plugins")
    recovery_plugin_dir = os.path.join(plugins_dir, PLUGIN_NAME)
    edit_cuberite_config.remove_plugin(settings_ini, "logging")
    edit_cuberite_config.add_plugin(settings_ini, PLUGIN_NAME)
    if not os.path.isdir(recovery_plugin_dir):
        shutil.copytree(
            os.path.join(repo_home, "server/cuberite_plugins", PLUGIN_NAME), recovery_plugin_dir
        )

    # Read logging.bin to get chunks available, and rewrite recovery plugin
    chunks = get_chunks_avail(old_workdir)
    chunks_lua = tuple_list_to_lua(chunks)
    with open(os.path.join(recovery_plugin_dir, "recover_initial.lua"), "r") as f:
        recovery_lua = f.read()
    recovery_lua = recovery_lua.replace("__CHUNKS_TO_LOAD__", chunks_lua)
    with open(os.path.join(recovery_plugin_dir, "recover_initial.lua"), "w") as f:
        f.write(recovery_lua)

    # Start craftassist_cuberite and wait until the plugin kills it
    p = subprocess.Popen([repo_home + "/server/craftassist_cuberite/Server/Cuberite"], cwd=workdir)
    p.wait()

    # Return folder containing region files
    return os.path.join(workdir, "world/region")


def get_chunks_avail(logdir):
    chunks = []

    class ChunkAvailLogReader(BaseLogReader):
        def on_chunk_available(self, buf_start, hid, cx, cz):
            chunks.append((cx, cz))

    ChunkAvailLogReader(logdir).start()
    return chunks


def tuple_list_to_lua(tuple_list):
    """Given a list of tuples, return a lua table of tables"""

    def table(it):
        return "{" + ",".join(map(str, it)) + "}"

    return table(table(t) for t in tuple_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("workdir")
    args = parser.parse_args()

    recover_initial_blockmap(args.workdir)
