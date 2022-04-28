# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

__author__ = "Tingfan Wu"
__copyright__ = "2022, Facebook"


setup(
    name="eyehandcal",
    author="Tingfan Wu",
    author_email="tingfan@fb.com",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=["src/eyehandcal/scripts/collect_data_and_cal.py", "src/eyehandcal/scripts/record_calibration_points.py"],
)
