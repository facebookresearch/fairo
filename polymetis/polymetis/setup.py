# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages

scripts = []
for here, dirs, files in os.walk("python/scripts"):
    for file in files:
        if (file.endswith(".py")):
            scripts.append(os.path.join(here, file))

packages = find_packages(where="python") + find_namespace_packages(
    include=["hydra_plugins.*"], where="python"
)

setup(
    name="polymetis",
    version="0.2",
    packages=packages,
    package_dir={"": "python"},
    include_package_data=True,
    scripts=scripts,
    install_requires=["alephzero"],
)
