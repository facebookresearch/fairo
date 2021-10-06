import time

import numpy as np
import sophus as sp
import torch
import cv2
import pyrealsense2 as rs

from polymetis import RobotInterface, GripperInterface
import arucoX as ax


# Markers
TABLE_MARKER_ID = 2
TABLE_MARKER_LENGTH = 0.1
OBJECT_MARKER_ID = 3
OBJECT_MARKER_LENGTH = 0.05

# Calib params
NUM_CALIB_SAMPLES = 5
CALIB_SAMPLE_INTERVAL = 0.5

# Grasp params
ORIGIN_EE_POS = [
    0.2892,
    -0.2832,
    0.1746,
]  # EE position after guiding the gripper to right above the table marker
PREGRASP_HEIGHT = 0.05


class Robot:
    def __init__(self, ip, robot_port, gripper_port):
        # Initialize arm
        self.arm = RobotInterface(ip_address=ip, port=robot_port)

        # Initialize gripper
        self.gripper = GripperInterface(ip_address=ip, port=gripper_port)

        # Coordinate transform
        self.scene2arm_xyz = np.array(ORIGIN_EE_POS)
        self.pregrasp_offset = np.array([0, 0, PREGRASP_HEIGHT])

        # Update home pose
        home_pose = self.arm.home_pose
        home_pose[-1] = home_pose[-1] + np.pi / 2.0
        self.arm.set_home_pose(home_pose)

    def reset(self):
        # Open gripper
        self.gripper.move(width=0.1, speed=0.1)

        # Reset arm pose
        self.arm.go_home()

    def grasp(self, marker_pose):
        # Compute grasp pose in arm EE coordinate system
        pregrasp_pos = self.scene2arm_xyz + marker_pose.translation()
        grasp_pos = pregrasp_pos - self.pregrasp_offset

        # Move to pregrasp
        self.arm.set_ee_pose(torch.Tensor(pregrasp_pos), time_to_go=3.0)

        # Perform grasp
        self.arm.set_ee_pose(torch.Tensor(grasp_pos), time_to_go=1.5)
        self.gripper.grasp(width=0.05, speed=0.1, force=0.1)

        # Lift
        self.arm.set_ee_pose(torch.Tensor(pregrasp_pos), time_to_go=1.0)


class RealSenseCamera:
    """ Wrapper that implements boilerplate code for RealSense cameras """

    def __init__(self):
        # Start stream
        print("Connecting to RealSense camera...")
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start()
        print("Connected.")

        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(60):
            self.pipe.wait_for_frames()

    def get_intrinsics(self):
        stream = self.profile.get_streams()[1]  # 0: depth, 1: color
        intrinsics = stream.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = intrinsics.fx
        camera_matrix[1, 1] = intrinsics.fy
        camera_matrix[0, 2] = intrinsics.ppx
        camera_matrix[1, 2] = intrinsics.ppy

        dist_coeffs = np.array(intrinsics.coeffs)

        return camera_matrix, dist_coeffs

    def get_image(self):
        frame = self.pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_img = np.asanyarray(color_frame.get_data())

        return color_img[:, :, ::-1]  # BGR to RGB


if __name__ == "__main__":
    # Initialize robot
    robot = Robot(ip="localhost", robot_port="50051", gripper_port="50052")

    # Initialize camera
    camera = RealSenseCamera()
    matrix, dist_coeffs = camera.get_intrinsics()

    # Initialize camera module & scene module
    c = ax.CameraModule()
    c.set_intrinsics(matrix=matrix, dist_coeffs=dist_coeffs)
    scene = ax.Scene(cameras=[c])

    # Register markers
    scene.register_marker_size(TABLE_MARKER_ID, TABLE_MARKER_LENGTH)
    scene.register_marker_size(OBJECT_MARKER_ID, OBJECT_MARKER_LENGTH)
    scene.set_origin_marker(TABLE_MARKER_ID)

    #######################
    # Calibrate scene
    #######################
    # Capture images
    for _ in range(NUM_CALIB_SAMPLES):
        img = camera.get_image()
        scene.add_snapshot([img])
        time.sleep(CALIB_SAMPLE_INTERVAL)

    # Calibrate
    scene.calibrate_extrinsics()

    # Optional: visualize scene
    # scene.visualize()

    #######################
    # Grasping experiment
    #######################
    robot.reset()
    try:
        while True:
            # The operator should place the object at an arbitrary position now
            input("Press enter to perform grasp. ")

            # Pose estimation
            img = camera.get_image()
            pose = scene.estimate_marker_pose([img], marker_id=OBJECT_MARKER_ID)

            # Perform grasp
            print(f"Detected object position: {pose.translation()}")
            if pose is not None:
                robot.grasp(pose)
                time.sleep(1)
                robot.reset()
            else:
                print("Object marker not detected. Try again.")

    except KeyboardInterrupt:
        print("Terminated by user.")
