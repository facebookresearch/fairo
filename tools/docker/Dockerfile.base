# Copyright (c) Facebook, Inc. and its affiliates.
# This is pre-built and is imported in ./Dockerfile as base
FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get install -y software-properties-common  # for add-apt-repository
RUN add-apt-repository ppa:git-core/ppa -y
RUN apt-get -y update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y \
	curl \
	lld \
        g++ \
        clang-format \
        git \
	time \
        htop \
        libboost-all-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        make \
        zlib1g-dev \
        libblas-dev \
        liblapack-dev \
        wget \
        ;

SHELL ["/bin/bash", "-c"]
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
SHELL ["/bin/bash", "-c"]

RUN conda install mamba setuptools cmake -y -c conda-forge

ADD ./conda.txt /conda.txt
ADD ./requirements.txt /pip.txt

RUN mamba create -n droidlet python=3.8 -c conda-forge -y && \
    conda init bash && \
    source ~/.bashrc && \
    source activate /root/miniconda3/envs/droidlet && \
    pip install -r /pip.txt && \
    conda clean -a -y && \
    pip cache purge

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get -y install nodejs

# Install Git LFS
COPY gitlfs_install.sh /gitlfs_install.sh
RUN add-apt-repository ppa:git-core/ppa \
        && bash /gitlfs_install.sh \
        && apt-get install -y git-lfs \
        && git lfs install


# get latest ccache instead of system-packaged one
RUN apt remove -y ccache
RUN apt install -y libzstd-dev
RUN cd /tmp && curl -L https://github.com/ccache/ccache/releases/download/v4.3/ccache-4.3.tar.gz -o ccache-4.3.tar.gz && \
    tar -xvf ccache-4.3.tar.gz && \
    cd ccache-4.3 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make && \
    make install && \
    ln -s ccache /usr/local/bin/gcc && \
    ln -s ccache /usr/local/bin/g++ && \
    ln -s ccache /usr/local/bin/cc && \
    ln -s ccache /usr/local/bin/c++
ENV PATH "/usr/local/bin:$PATH"
ENV CCACHE_DIR "/cache/.ccache"
RUN mkdir -p /cache/.ccache
