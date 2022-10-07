import logging
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import Pose

from iphone_reader import iPhoneReader

log = logging.getLogger(__name__)

IPHONE_OFFSET = [0, 0.1]  # XY offset in base frame
IPHONE_ROT = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
]


class Record3dSLAM:
    def __init__(self):
        self.reader = iPhoneReader(retrieve_imgs=False)

        self.base_pose = np.zeros(3)
        self.pose_lock = threading.Lock()

        # Init publishers
        self.c_pose_pub = rospy.Publisher("r3d_slam/camera_pose", Pose, queue_size=1)
        self.b_pose_pub = rospy.Publisher("r3d_slam/base_pose", Pose, queue_size=1)

        # Compute transformation between base & iphone
        self.T_b2i = np.eye(4)
        self.T_b2i[:2, 3] = np.array(IPHONE_OFFSET)
        self.T_b2i[:3, :3] = np.array(IPHONE_ROT)
        self.T_i2b = np.linalg.pinv(self.T_b2i)

        # Init ros node
        rospy.init_node("iphone_slam")

    def _on_new_frame(self, frame):
        # Extract camera pose
        pose_i = frame.pose_mat

        # Compute & update base pose
        pose_b = self.T_b2i @ pose_i @ self.T_i2b
        pose_pos = pose_b[:3, 3]
        pose_ori = R.from_matrix(pose_b[:3, :3])
        pose_quat = pose_ori.as_quat()

        with self.pose_lock:
            self.base_pose[:2] = pose_pos[:2]
            self.base_pose[2] = pose_ori.as_rotvec()[2]

        # Publish to rostopic
        camera_pose_ros = Pose()
        camera_pose_ros.position.x = frame.pose_pos[0]
        camera_pose_ros.position.y = frame.pose_pos[1]
        camera_pose_ros.position.z = frame.pose_pos[2]
        camera_pose_ros.orientation.x = frame.pose_quat[0]
        camera_pose_ros.orientation.y = frame.pose_quat[1]
        camera_pose_ros.orientation.z = frame.pose_quat[2]
        camera_pose_ros.orientation.w = frame.pose_quat[3]
        self.c_pose_pub.publish(camera_pose_ros)

        base_pose_ros = Pose()
        base_pose_ros.position.x = self.base_pose[0]
        base_pose_ros.position.y = self.base_pose[1]
        base_pose_ros.position.z = 0.0
        base_pose_ros.orientation.x = pose_quat[0]
        base_pose_ros.orientation.y = pose_quat[1]
        base_pose_ros.orientation.z = pose_quat[2]
        base_pose_ros.orientation.w = pose_quat[3]
        self.b_pose_pub.publish(base_pose_ros)

    def get_base_pose(self):
        with self.pose_lock:
            pose = self.base_pose.copy()
        return pose

    def start(self):
        self.reader.start(frame_callback=self._on_new_frame)
        rospy.spin()


if __name__ == "__main__":
    slam = Record3dSLAM()
    slam.start()
