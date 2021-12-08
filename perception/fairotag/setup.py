#!/usr/bin/env python
######################################################################
# \file setup.py
# \author Austin Wang
#######################################################################
from setuptools import setup, find_packages

__author__ = "Austin Wang"
__copyright__ = "2020, Facebook"

install_requires = ["numpy", "matplotlib", "opencv-contrib-python", "scipy", "sophuspy"]

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
