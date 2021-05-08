#!/usr/bin/env python
import os
import io
import re
import shutil
import sys
from setuptools import setup, find_packages

readme = open("README.md").read()

setup(
    # Metadata
    name="droidlet",
    author="Facebook AI",
    author_email="apratik@fb.com",
    description="image and video datasets and models for torch deep learning",
    long_description=readme,
    license="MIT",
    # Package info
    packages=find_packages(exclude=("test", "craftassist/test", "locobot/test")),
    zip_safe=True,
)
