#!/usr/bin/env python
######################################################################
# \file setup.py
# \author Austin Wang
#######################################################################
from setuptools import setup

__author__ = "Austin Wang"
__copyright__ = "2022, Facebook"

install_requires = [
    "numpy",
    "scipy",
    "record3d",
    "pytest",
]

setup(
    name="iphone_reader",
    author="Austin Wang",
    author_email="wangaustin@fb.com",
    version=1.0,
    packages=["iphone_reader"],
    package_dir={"": "src"},
    install_requires=install_requires,
    zip_safe=False,
)
