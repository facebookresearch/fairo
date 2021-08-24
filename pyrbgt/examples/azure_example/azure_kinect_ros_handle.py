import numpy as np
import cv2
import os

from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import CameraInfo, Image
import threading
import copy
import message_filters
import time

from pyrbgt import ImageHandle, Intrinsics


class AzureKinectRosHandle(ImageHandle):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.camera_info_lock = threading.RLock()
        self.camera_img_lock = threading.RLock()
        self.rgb_img = None
        self.depth_img = None
        self.camera_info = None
        self.camera_P = None
        rospy.Subscriber(
            "/rgb/camera_info", CameraInfo, self._camera_info_callback,
        )

        rgb_topic = "/rgb/image_raw"
        self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_topic = "/depth_to_rgb/image_raw"
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        img_subs = [self.rgb_sub, self.depth_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            img_subs, queue_size=10, slop=0.2
        )
        self.sync.registerCallback(self._sync_callback)
        rospy.sleep(1)

    def _sync_callback(self, rgb, depth):
        self.camera_img_lock.acquire()
        try:
            self.rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
        self.camera_img_lock.release()

    def _camera_info_callback(self, msg):
        self.camera_info_lock.acquire()
        self.camera_info = msg
        self.camera_P = np.array(msg.P).reshape((3, 4))
        self.camera_info_lock.release()

    def get_intrinsics(self):
        intrinsics = Intrinsics()
        if self.camera_P is not None and self.camera_info is not None:
            intrinsics.fu = float(self.camera_P[0][0])
            intrinsics.fv = float(self.camera_P[1][1])
            intrinsics.ppu = float(self.camera_P[0][2])
            intrinsics.ppv = float(self.camera_P[1][2])
            intrinsics.width = self.camera_info.width
            intrinsics.height = self.camera_info.height
        return intrinsics

    def get_image(self):
        self.camera_img_lock.acquire()
        rgb = copy.deepcopy(self.rgb_img)
        self.camera_img_lock.release()
        return rgb
