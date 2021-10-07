# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages

install_requires = [
    "hydra-core==1.0.6",
    "numpy>=1.18.0",
    "pandas",
    "plotly",
    "tqdm",
    "dash",
    "grpcio",
    "scipy >= 1.6.0",
    "pytest >= 6.1.2",
    "pytest-benchmark >= 3.4.1",
    "Sphinx >= 3.5.4",
    "sphinx-book-theme",
    "breathe >= 4.29.1",
    "myst-parser >= 0.13.7",
    "pybullet==3.1.7",
    "pyserial",
    "pymodbus",
]

script_dir = "python/scripts"
scripts = [
    os.path.join(script_dir, file)
    for file in os.listdir(script_dir)
    if file.endswith(".py")
]

packages = find_packages(where="python") + find_namespace_packages(
    include=["hydra_plugins.*"], where="python"
)

setup(
    name="polymetis",
    version="0.2",
    packages=packages,
    package_dir={"": "python"},
    install_requires=install_requires,
    include_package_data=True,
    scripts=scripts,
)
