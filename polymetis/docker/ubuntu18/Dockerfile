FROM ubuntu:18.04

RUN apt update -y && apt upgrade -y

# miniconda

RUN apt install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# install essentials

RUN apt install -y build-essential cmake libtinfo-dev lsb-release libmpfr-dev libedit-dev freeglut3 \
    freeglut3-dev libxmu-dev libxmu-headers libhiredis-dev unzip libnuma-dev

# install nvidia drivers
RUN DEBIAN_FRONTEND=noninteractive apt install -y nvidia-driver-460-server
