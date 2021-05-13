# Copyright (c) Facebook, Inc. and its affiliates.
FROM shaomai/craftassist:latest as base

COPY mcdeploy.key /mcdeploy.key
COPY AWS_S3_ACCESS_KEY_ID.txt /AWS_S3_ACCESS_KEY_ID.txt
COPY AWS_S3_SECRET_ACCESS_KEY.txt /AWS_S3_SECRET_ACCESS_KEY.txt
ARG current_branch

# Clone/make repo
ENV GIT_SSH_COMMAND "ssh -i /mcdeploy.key -o StrictHostKeyChecking=no"
RUN git clone --recursive git@github.com:facebookresearch/droidlet.git
WORKDIR droidlet
RUN git checkout ${current_branch}

RUN pip3 install -r requirements.txt
RUN pip3 install -U setuptools

RUN tools/data_scripts/try_download.sh craftassist

RUN mkdir -p /cache/.ccache
RUN (AWS_ACCESS_KEY_ID=$(cat /AWS_S3_ACCESS_KEY_ID.txt) AWS_SECRET_ACCESS_KEY=$(cat /AWS_S3_SECRET_ACCESS_KEY.txt) curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/client_ccache.tar.gz -o /client_ccache.tar.gz -f && tar -xvf /client_ccache.tar.gz -C /cache) || echo "Pulling ccache from S3"
RUN export CCACHE_DIR="/cache/.ccache"
RUN git submodule update --init --recursive
RUN cd craftassist && make && cd ..
RUN tar -cvzf client_ccache.tar.gz /cache/.ccache
RUN AWS_ACCESS_KEY_ID=$(cat /AWS_S3_ACCESS_KEY_ID.txt) AWS_SECRET_ACCESS_KEY=$(cat /AWS_S3_SECRET_ACCESS_KEY.txt) aws s3 cp client_ccache.tar.gz s3://craftassist/pubr/client_ccache.tar.gz

RUN npm install -g yarn
RUN cd /droidlet/dldashboard/web && yarn install

RUN mkdir workdir
WORKDIR workdir

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY run.fromenv.sh /run.fromenv.sh

EXPOSE 3000
EXPOSE 2556
EXPOSE 2557
EXPOSE 25565
EXPOSE 9000
EXPOSE 5000

ENTRYPOINT ["/run.fromenv.sh"]  # uses $RUN_SH_GZ_B64
