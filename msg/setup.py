from setuptools import setup, find_packages


__author__ = "Leonid Shamis"
__copyright__ = "2022, Meta"


install_requires = [
    "pycapnp",
]


setup(
    name="fairomsg",
    author="Leonid Shamis",
    author_email="lshamis@fb.com",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
