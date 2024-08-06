# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import polymetis


PKG_ROOT_DIR = polymetis.__path__[0]
DATA_DIR = os.path.join(PKG_ROOT_DIR, "data")
BUILD_DIR = os.path.abspath(os.path.join(PKG_ROOT_DIR, "..", "..", "build"))


def get_full_path_to_urdf(path: str):
    """Gets the absolute path to a relative path of :code:`DATA_DIR`."""
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(DATA_DIR, path))
    assert os.path.exists(path), f"Invalid robot_description_path: {path}"
    return path


def which(program: str):
    """Equivalent of `which <https://en.wikipedia.org/wiki/Which_(command)>`_ program.

    Taken from https://stackoverflow.com/a/377028
    Returns equivalent of $(which program), or None
    if unable to find it.

    Args:
        program: name of the executable to find.
    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
