import logging
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import Pose

from record3d import Record3DStream

log = logging.getLogger(__name__)

IPHONE_OFFSET = [0, 0.1]  # XY offset in base frame


class Record3dSLAM:
    def __init__(self):
        self.session = None

        self.base_pose = np.zeros(3)
        self.pose_lock = threading.Lock()

        self.c_pose_pub = rospy.Publisher("r3d_slam/camera_pose", Pose, queue_size=1)
        self.b_pose_pub = rospy.Publisher("r3d_slam/base_pose", Pose, queue_size=1)

        # Compute transformation between base & iphone
        self.T_b2i = np.eye(4)
        self.T_b2i[:2, 3] = np.array(IPHONE_OFFSET)
        self.T_b2i[:3, :3] = R.from_rotvec([np.pi / 2.0, 0.0, 0.0]).as_matrix()
        self.T_i2b = np.linalg.pinv(self.T_b2i)

    def on_new_frame(self):
        # Extract camera pose
        # NOTE: quat & pos in world frame: camera_pose.[qx|qy|qz|qw|tx|ty|tz])
        camera_pose = self.session.get_camera_pose()
        pose_i = np.eye(4)
        pose_i[:3, 3] = np.array([camera_pose.tx, camera_pose.ty, camera_pose.tz])
        pose_i[:3, :3] = R.from_quat(
            [camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw]
        ).as_matrix()

        # Compute & update base pose
        pose_b = self.T_b2i @ pose_i @ self.T_i2b

        with self.pose_lock:
            self.base_pose[:2] = pose_b[:2, 3]
            self.base_pose[2] = R.from_matrix(pose_b[:3, :3]).as_rotvec()[2]

        # Publish to rostopic
        camera_pose_ros = Pose()
        camera_pose_ros.linear.x = camera_pose.tx
        camera_pose_ros.linear.y = camera_pose.ty
        camera_pose_ros.linear.z = camera_pose.tz
        camera_pose_ros.angular.x = camera_pose.qx
        camera_pose_ros.angular.y = camera_pose.qy
        camera_pose_ros.angular.z = camera_pose.qz
        camera_pose_ros.angular.w = camera_pose.qw
        self.c_pose_pub.publish(camera_pose_ros)

        base_pose_ros = Pose()
        base_pose_ros.linear.x = self.base_pose[0]
        base_pose_ros.linear.y = self.base_pose[1]
        base_pose_ros.angular.z = self.base_pose[2]
        self.b_pose_pub.publish(base_pose_ros)

    def on_stream_stopped(self):
        log.info("Stream stopped")

    def connect_to_device(self, dev_idx=0):
        log.info("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        log.info("{} device(s) found".format(len(devs)))
        for dev in devs:
            log.info("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(dev_idx)
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_base_pose(self):
        with self.pose_lock:
            pose = self.base_pose.copy()
        return pose


if __name__ == "__main__":
    slam = Record3dSLAM()
    slam.connect_to_device(0)
