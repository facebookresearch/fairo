from setuptools import find_packages
from setuptools import setup


setup(
    name="fbrp",
    version="0.0.8",
    author="Leonid Shamis",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["*"],
    ),
    python_requires=">=3.7",
    install_requires=[
        "aiodocker>=0.21.0",
        "alephzero>=0.3.11",
        "docker>=5.0.0",
        "psutil>=5.8.0",
        "pyyaml>=6.0",
        "six>=1.16.0",
    ],
)
