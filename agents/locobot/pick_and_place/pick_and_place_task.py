import cv2
import numpy as np
import rospy

from constants import coco_categories

# ----------------------
# Robot planning tools
# TODO(cpaxton): move these all into fairo
from home_robot.hardware.stretch_ros import HelloStretchROSInterface
from home_robot.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.motion.robot import STRETCH_STANDOFF_DISTANCE
from home_robot.ros.path import get_package_path
from home_robot.ros.camera import RosCamera
from home_robot.utils.pose import to_pos_quat
from home_robot.utils.numpy import to_npy_file
from home_robot.ros.grasp_helper import GraspClient as RosGraspClient
import home_robot.utils.image as hrimg
import trimesh
import trimesh.transformations as tra


"""
Things to install:
    pip install rospkg
    pip install pybullet  # Used for kinematics
    pip install trimesh  # used for motion planning
    # tracikpy for inverse kinematics
"""


class PickAndPlaceTask:
    """ Create pick and place task that integrates with navigation and planning """
    def __init__(self, mover):
        self.nav = mover.nav    # Point goal nav + semantic exploration
        self.slam = mover.slam  # Semantic and obstacle map + last frame
        self.bot = mover.bot    # Main robot class

        self.num_segment_attempts = 100
        self.num_grasp_attempts = 10
        self.min_obj_pts = 100

        # ROS connection into the robot
        # TODO: this needs to be replaced by code that exists in them over
        visualize = False  # Debugging flag, renders kinematics in pybullet
        self.manip = HelloStretchROSInterface(visualize_planner=visualize,
                                              root=get_package_path(),
                                              init_cameras=False,  # ROS camera intialization
                                              )
        self.home_q = STRETCH_HOME_Q.copy()
        # Get the kinematic model for the manipulator in case we end up needing it
        self.model = self.manip.get_model()
        # Look ahead to start out with
        self.manip.look_ahead()
        self.grasp_client = RosGraspClient()

        # Parameters for configuring pick and place motions
        self.exploration_method = "learned"

        # This is for enabling integration with home-robot grasping code. If this is disabled, we
        # do not need to run any base motion related commands.
        self.navigation_enabled = False

    def pick_and_place(self, start_receptacle: str, object: str, end_receptacle: str):
        """
        End-to-end pick and place with semantic exploration and mobile
        object grasping and placing.

        Arguments:
            start_receptacle: category of receptacle the object is initially on
            object: category of object to grasp
            end_receptacle: category of receptacle the object ends up on
        """
        print(f"Starting pick {object} from {start_receptacle} and place it "
              f"on {end_receptacle}")
        assert start_receptacle in [
            "chair", "couch", "bed", "toilet", "dining-table", "sink"]
        assert end_receptacle in [
            "chair", "couch", "bed", "toilet", "dining-table", "sink"]
        assert object in ["cup", "bottle"]

        if self.navigation_enabled:
            # we would use the navigation service for semantic exploration like below
            self.nav.go_to_object(
                object_goal=start_receptacle,
                episode_id=f"go_to_{start_receptacle}",
                exploration_method=self.exploration_method,
                debug=false,
                visualize=true,
                max_steps=400,
                start_with_panorama=true,
            )
        # Pass object into picking code
        self.pick(object)
        
    def pick(self, object: str):
        """
        Mobile grasping of an object category present in the last frame.
        
        Arguments:
            object: category of object to grasp
        """
        print(f"Starting pick {object}")
        assert object in ["cup", "bottle"]
        category_id = coco_categories[object]

        # Look at end effector and wait long enough that we have a new observation
        self.manip.stow()
        rospy.sleep(0.5)
        self.manip.look_at_ee()
        rospy.sleep(0.5)

        print("Here is the data you have available about the last time the "
              "semantic map got updated, which might be slightly stale if the "
              "robot has been moving:")
        
        grasp_attempts_made = 0
        for attempt in range(self.num_segment_attempts):
            info = self.slam.get_last_position_vis_info()
            flat_pcd = info["pcd"]
            flat_object_mask = info["semantic_frame"][:, category_id]
            image_object_mask = info["unfiltered_semantic_frame"][:, :, category_id]
            semantic_frame = info["semantic_frame_vis"]
            obstacle_map = info["semantic_map"][0]
            object_map = info["semantic_map"][4 + category_id]
            orig_rgb = info['rgb']

            if attempt == 0:
                print(list(info.keys()))
                print("flat_pcd.shape", flat_pcd.shape)
                print("flat_object_mask.shape", flat_object_mask.shape)
                print("image_object_mask.shape", image_object_mask.shape)
                print("obstacle_map.shape", obstacle_map.shape)
                print("object_map.shape", object_map.shape)
                print()

                cv2.imwrite("semantic_frame.png", semantic_frame)
                cv2.imwrite("image_object_mask.png", (image_object_mask * 255).astype(np.uint8))
                cv2.imwrite("obstacle_map.png", (obstacle_map * 255).astype(np.uint8))
                cv2.imwrite("object_map.png", (object_map * 255).astype(np.uint8))

                print("Here is how to transform robot coordinates to map coordinates:")

                pose_of_last_map_update = info["pose"]
                pose_in_map_coordinates = self.slam.robot2map(
                    pose_of_last_map_update[:2])

                print("pose_of_last_map_update", pose_of_last_map_update)
                print("curr_pose_in_map_coordinates", pose_in_map_coordinates)
                print()

            num_obj_pts = np.sum(image_object_mask)
            print(attempt, "Detected this many object points:", num_obj_pts)
            if num_obj_pts < self.min_obj_pts:
                print("Too few object points; trying to segment again...")
                continue

            print("Attempting to grasp...")
            predicted_grasps = self.grasp_client.request(flat_pcd,
                                                         orig_rgb,
                                                         flat_object_mask,
                                                         frame="rgb_optical_frame")
            print("options =", [(k, v[-1].shape) for k, v in predicted_grasps.items()])
            predicted_grasps, scores = predicted_grasps[0]
        else:
            print("FAILED TO GRASP! Could not find the object.")


    def place(self, end_receptacle):
        """Mobile placing of the object picked up."""
        print("Starting place")
        if self.navigation_enabled:
            self.nav.go_to_object(
                object_goal=end_receptacle,
                episode_id=f"go_to_{end_receptacle}",
                exploration_method=self.exploration_method,
                debug=false,
                visualize=true,
                max_steps=400,
                start_with_panorama=true,
            )



def test_pick_place(mover, value):
    start_receptacle, object, end_receptacle = [x.strip() for x in value.split("_")]
    print("Start receptacle =", start_receptacle)
    print("Object           =", object)
    print("End receptacle   =", end_receptacle)
    print(f"action: PICK_AND_PLACE {object} from {start_receptacle} to {end_receptacle}")
    task = PickAndPlaceTask(mover)
    task.pick_and_place(start_receptacle, object, end_receptacle)


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP.",
        type=str,
        default="192.168.0.49",
    )
    parser.add_argument(
        "--backend",
        help="Which backend to use: habitat, hellorobot",
        type=str,
        default='hellorobot',
    )
    args = parser.parse_args()
    
    ip = args.ip
    backend = args.backend
    
    print("Connecting to robot at ip: ", ip)

    if backend == 'habitat':
        from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
        mover = LoCoBotMover(ip=ip, backend='habitat')
    elif backend == 'hellorobot':
        from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover
        mover = HelloRobotMover(ip=ip)
    print("Mover is ready to be operated")

    log_settings = {
        "image_resolution": 512,  # pixels
        "image_quality": 10,  # from 10 to 100, 100 being best
    }

    all_points = None
    all_colors = None
    first = True
    prev_stg = None
    path_count = 0

    start_time = time.time_ns()
    fps_freq = 1 # displays the frame rate every 1 second
    counter = 0
    if backend == 'habitat':
        mover.bot.set_pan(0.0)
        # mover.bot.set_tilt(-1.5)
    else: # hellorobot
        mover.bot.set_pan(0.0)
        # mover.bot.set_tilt(-1.05)

    rospy.init_node('test_pick_place')
    test_pick_place(mover, "chair_cup_chair")