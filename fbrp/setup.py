from setuptools import find_packages
from setuptools import setup


setup(
    name="facebook_robotics_platform",
    version="0.0.1",
    author="Leonid Shamis",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["*"],
    ),
    python_requires=">=3.7",
    install_requires=[
        "aiodocker",
        "docker",
        "six",
        "alephzero==v0.3",
    ],
)
