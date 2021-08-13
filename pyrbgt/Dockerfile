FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

RUN apt update && \
  DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    libeigen3-dev \
    libglew-dev \
    libglfw3-dev \
    libopencv-dev \
    python3-dev \
    python3-pip && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
  torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install \
  pytorch3d==0.5.0 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt190/download.html

COPY . /pyrbgt

RUN python3 -m pip install -r /pyrbgt/requirements.txt

RUN cd /pyrbgt && python3 -m pip install .
