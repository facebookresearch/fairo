from setuptools import setup, find_packages

"""
Wrapper package defining the interface for grasping primitives in Fairo.
"""

__author__ = "Yixin Lin"
__copyright__ = "2022, Meta"


install_requires = [
    "mrp",
    "graspnetAPI",
    "open3d",
    "fairomsg",
    "realsense_wrapper",
    "ikpy",  # TODO: switch to better IK lib
]

setup(
    name="polygrasp",
    author="Yixin Lin",
    author_email="yixinlin@fb.com",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    scripts=["scripts/run_grasp.py"],
    install_requires=install_requires,
)
