#!/usr/bin/env python
######################################################################
# \file setup.py
# \author Austin Wang
#######################################################################
from setuptools import setup, find_packages

__author__ = "Austin Wang"
__copyright__ = "2022, Facebook"

install_requires = [
    "gtsam==4.1.1",
    "numpy",
    "matplotlib",
    "opencv-contrib-python<=4.7.0", # works with cv2.aruco API < 4.7
    "scipy",
    "sophuspy",
]

setup(
    name="fairotag",
    author="Austin Wang",
    author_email="wangaustin@fb.com",
    version=1.0,
    packages=["fairotag"],
    package_dir={"": "python"},
    install_requires=install_requires,
    zip_safe=False,
)
