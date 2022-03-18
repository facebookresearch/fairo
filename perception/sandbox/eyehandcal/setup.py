from setuptools import setup, find_packages

__author__ = "Tingfan Wu"
__copyright__ = "2022, Facebook"


setup(
    name="eyehandcal",
    author="Tingfan Wu",
    author_email="tingfan@fb.com",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=["scripts/collect_data_and_cal.py"]
)