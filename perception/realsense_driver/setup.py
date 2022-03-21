#!/usr/bin/env python
from setuptools import setup, find_packages

__author__ = "Austin Wang"
__copyright__ = "2021, Facebook"

install_requires = ["numpy", "pyrealsense2"]

setup(
    name="realsense_wrapper",
    author="Austin Wang",
    version=1.0,
    packages=["realsense_wrapper"],
    package_dir={"": "python"},
    install_requires=install_requires,
    zip_safe=False,
)
