from setuptools import setup, find_packages

"""
Wrapper package defining the interface for grasping primitives in Fairo.
"""

__author__ = "Yixin Lin"
__copyright__ = "2022, Meta"


setup(
    name="polygrasp",
    author="Yixin Lin",
    author_email="yixinlin@fb.com",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=["scripts/run_grasp.py"],
)
