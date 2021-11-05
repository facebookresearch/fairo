FROM moveit/moveit:noetic-release

RUN apt update && \
    apt install -y \
        python3-pip \
        ros-noetic-rospy-message-converter
RUN python3 -m pip install alephzero>=v0.3

COPY panda_planner.py /

CMD [ "python3", "panda_planner.py" ]
