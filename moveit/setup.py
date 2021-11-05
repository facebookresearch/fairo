# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import setup
from setuptools import find_packages

install_requires = ["pybullet", "alephzero"]

setup(
    name="moveit_bridge",
    version="0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=install_requires,
)
