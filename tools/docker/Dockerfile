# Copyright (c) Facebook, Inc. and its affiliates.
FROM shaomai/craftassist:1 as base

ARG current_branch
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# Clone/make repo
RUN time git clone --recurse-submodules="./droidlet" --shallow-submodules --depth 1 --branch ${current_branch} --jobs 5 https://github.com/facebookresearch/fairo
WORKDIR fairo

RUN source activate /root/miniconda3/envs/droidlet
RUN mamba install python=3.8 --file conda.txt -c conda-forge  -c pytorch -c nvidia
RUN pip3 install -r requirements.txt
RUN pip3 install -r agents/craftassist/requirements.txt

# use the much faster lld linker
RUN rm -f /usr/bin/ld && ln -s /usr/bin/ld.lld /usr/bin/ld

RUN ((curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/client_ccache.tar.gz -o /client_ccache.tar.gz -f && \
      tar -xvf /client_ccache.tar.gz -C /cache/.ccache) \
      || echo "Failed pulling ccache from S3") && \
    ccache -s && \
    ls -alh /client_ccache.tar.gz && \
    ls -alh /cache/.ccache && \
    cd droidlet/lowlevel/minecraft && \
    CC="/usr/local/bin/gcc" CXX="/usr/local/bin/g++" make && \
    cd ../../../ && \
    tar -cvzf client_ccache.tar.gz -C /cache/.ccache . && \
    ls -alh client_ccache.tar.gz && \
    aws s3 cp client_ccache.tar.gz s3://craftassist/pubr/client_ccache.tar.gz --acl public-read && \
    ls -alh /cache/.ccache && \
    ccache -s && \
    rm -rf client_ccache.tar.gz /cache/.ccache /client_ccache.tar.gz

RUN cd /fairo && \
    python3 setup.py develop && \
    yes | python3 droidlet/tools/artifact_scripts/try_download.py --agent_name craftassist


RUN npm install -g yarn
RUN cd /fairo/droidlet/dashboard/web && yarn install

RUN mkdir workdir
WORKDIR workdir

ENV MCDASHBOARD_PORT 80

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

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
