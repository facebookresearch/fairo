from hydra.experimental import initialize, compose
from pyrbgt import RBGTTracker
from azure_kinect_ros_handle import AzureKinectRosHandle
import time

import rospy

if __name__ == "__main__":
    rospy.init_node("azure_kinect_camera")
    initialize(".")
    config = compose("ycb_banana.yaml")
    tracker = RBGTTracker(config.conf)
    image_handle = AzureKinectRosHandle()
    rospy.sleep(1)
    tracker.track(image_handle)
