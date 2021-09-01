# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import shlex
import subprocess
import numpy


def pkgconfig(info, lib):
    output = subprocess.check_output(["pkg-config", info, lib])
    return shlex.split(output.decode())


ext_modules = [
    Pybind11Extension(
        "rbgt_pybind",
        sorted(
            glob("rbgt_pybind/*.cpp") + glob("rbgt_pybind/src/*.cpp")
        ),  # Sort source files for reproducibility
        include_dirs=[
            "rbgt_pybind/third_party",
            "/usr/include/eigen3",
            "/usr/include/opencv4",
            numpy.get_include(),
        ],
        extra_compile_args=["-fopenmp"],
        extra_link_args=pkgconfig("--libs", "glew")
        + [
            "-fopenmp",
            "-lopencv_core",
            "-lopencv_highgui",
            "-lopencv_imgproc",
            "-lglfw",
            "-lstdc++fs",
        ],
    ),
]

setup(
    name="pyrbgt",
    version="0.0.1",
    author="PyRobot Team",
    url="https://github.com/facebookresearch/pyrobot.git",
    license="MIT",
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=["pyrbgt"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
