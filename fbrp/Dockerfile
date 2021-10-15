FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

ENV version=0.0.0

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    git \
    wget

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
    /bin/bash Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# RUN python3 -m pip install alephzero
RUN python3 -m pip install \
    alephzero@git+https://git@github.com/alephzero/py.git@v0.3

COPY . /fbrp
RUN cd /fbrp && \
    python3 -m pip install .
