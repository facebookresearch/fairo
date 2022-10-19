import logging
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import PoseStamped

from iphone_reader import Record3dReader

log = logging.getLogger(__name__)

IPHONE_OFFSET = [0, 0.1]  # XY offset in base frame
IPHONE_ROT = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
]


def generate_pose_stamped(pos, quat, time):
    pose = PoseStamped()

    pose.header.stamp = time

    pose.pose.position.x = pos[0]
    pose.pose.position.y = pos[1]
    pose.pose.position.z = pos[2]
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose


class Record3dSLAM:
    def __init__(self):
        self.reader = Record3dReader(retrieve_imgs=False)

        self.base_pose_2d = np.zeros(3)
        self.pose_lock = threading.Lock()

        # Init publishers
        self.c_pose_pub = rospy.Publisher("r3d_slam/camera_pose", PoseStamped, queue_size=1)
        self.b_pose_pub = rospy.Publisher("r3d_slam/base_pose", PoseStamped, queue_size=1)
        self.b2d_pose_pub = rospy.Publisher("r3d_slam/base_pose_2D", PoseStamped, queue_size=1)

        # Compute transformation between base & iphone
        self.T_b2i = np.eye(4)
        self.T_b2i[:2, 3] = np.array(IPHONE_OFFSET)
        self.T_b2i[:3, :3] = np.array(IPHONE_ROT)
        self.T_i2b = np.linalg.pinv(self.T_b2i)

        # Init ros node
        rospy.init_node("iphone_slam")

    def _on_new_frame(self, frame):
        ros_time = rospy.Time.now()

        # Extract camera pose
        pose_i = frame.pose_mat

        # Compute & update base pose
        pose_b = self.T_b2i @ pose_i @ self.T_i2b
        pose_pos = pose_b[:3, 3]
        pose_ori = R.from_matrix(pose_b[:3, :3])
        pose_quat = pose_ori.as_quat()

        with self.pose_lock:
            self.base_pose_2d[:2] = pose_pos[:2]
            self.base_pose_2d[2] = pose_ori.as_rotvec()[2]

        # Publish to rostopic
        camera_pose_ros = generate_pose_stamped(frame.pose_pos, frame.pose_quat, ros_time)
        base_pose_ros = generate_pose_stamped(pose_pos, pose_quat, ros_time)
        base_pose_2d_ros = generate_pose_stamped(
            [self.base_pose_2d[0], self.base_pose_2d[1], 0.0],
            R.from_rotvec([0.0, 0.0, self.base_pose_2d[2]]).as_quat(),
            ros_time,
        )

        self.c_pose_pub.publish(camera_pose_ros)
        self.b_pose_pub.publish(base_pose_ros)
        self.b2d_pose_pub.publish(base_pose_2d_ros)

    def get_base_pose_2d(self):
        with self.pose_lock:
            pose = self.base_pose_2d.copy()
        return pose

    def start(self):
        self.reader.start(frame_callback=self._on_new_frame)
        rospy.spin()


if __name__ == "__main__":
    slam = Record3dSLAM()
    slam.start()
