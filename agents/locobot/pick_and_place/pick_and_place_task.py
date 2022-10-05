import cv2
import numpy as np
import open3d as o3d
import rospy

from constants import coco_categories
from utils import get_pcd_in_cam

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

# For handling grasping
from home_robot.utils.pose import to_pos_quat
from home_robot.utils.numpy import to_npy_file

# for debugging
from geometry_msgs.msg import TransformStamped
from data_tools.point_cloud import show_point_cloud
from home_robot.ros.utils import ros_pose_to_transform, matrix_to_pose_msg


"""
Things to install:
    pip install rospkg
    pip install pybullet  # Used for kinematics
    pip install trimesh  # used for motion planning
    # tracikpy for inverse kinematics

On robot:
    ssh hello-robot@$ROBOT_IP  # ROBOT 1: 192.168.0.49, ROBOT 2: 192.168.0.48
    # in separate tabs
    roscore  # This is just to make development easier, not necessary
    roslaunch home_robot startup_stretch_hector_slam.launch start_rs_ros:=false
    droidlet && ./launch.sh --ros

Contact graspnet:
    cd ~/src/contact_graspnet; conda activate contact_graspnet_env; python contact_graspnet/graspnet_ros_server.py  --local_regions --filter_grasps

RVIZ: To visualize using rviz, on desktop (OPTIONAL STEP):
    # note that if you want to see the point cloud, we either need to:
    # (1) modify robot code to publish images to ros, OR
    # (2) run realsense briefly under ros:
    #     roslaunch home_robot startup_stretch_hector_slam.launch start_rs_ros:=true
    roslaunch home_robot visualization.launch

Finally, to run code:
    cd ~/src/fairo/agents/locobot/pick_and_place; conda activate droidlet
    python pick_and_place_task.py
"""


# Hard coded in remote_hello_robot_ros.py
BASE_HEIGHT = 0.091491526943


class PickAndPlaceTask:
    """ Create pick and place task that integrates with navigation and planning """
    def __init__(self, mover):
        self.nav = mover.nav    # Point goal nav + semantic exploration
        self.slam = mover.slam  # Semantic and obstacle map + last frame
        self.bot = mover.bot    # Main robot class
        self.intrinsic_mat = mover.cam.get_intrinsics()
        self.R_stretch_camera = tra.euler_matrix(0, 0, -np.pi/2)

        self.num_segment_attempts = 100
        self.num_grasp_attempts = 10
        self.min_obj_pts = 100
        self.min_predicted_grasps = 10

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
        self.grasp_client = RosGraspClient(flip_grasps=True)

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
                debug=False,
                visualize=True,
                max_steps=400,
                start_with_panorama=True,
            )
        # Pass object into picking code
        self.pick(object)

    def goto_static_grasp(self, grasps, scores=None, world_pcd=None, image_rgb=None, pause=False, debug=False):
        """
        Go to a grasp position, given a list of acceptable grasps.
        """
        if scores is None:
            scores = np.arange(len(grasps))
        q, _ = self.manip.update()

        # Some magic numbers here
        # This should correct for the length of the Stretch gripper and the gripper upon which
        # Graspnet was trained
        # STRETCH_STANDOFF_DISTANCE = 0.235
        grasp_offset = np.eye(4)
        grasp_offset[2, 3] = (-1 * STRETCH_STANDOFF_DISTANCE) + 0.05  # 0.12

        original_grasps = grasps.copy()
        for i, grasp in enumerate(grasps):
           grasps[i] = grasp @ grasp_offset

        base_theta_movements = []
        for grasp in grasps:
           grasp_pose = to_pos_quat(grasp)
           qi = self.model.static_ik(grasp_pose, q)
           if qi is not None:
               base_theta_movements.append(
                   np.abs(q[HelloStretchIdx.BASE_THETA] - qi[HelloStretchIdx.BASE_THETA])
               )
           else:
               base_theta_movements.append(np.inf)
        base_theta_movements = np.array(base_theta_movements)
        print("base_theta_movements: min, max, mean")
        print(
            base_theta_movements.min(),
            base_theta_movements.max(),
            base_theta_movements.mean()
        )

        for grasp, orig_grasp, score in sorted(zip(grasps, original_grasps, scores), key=lambda p: p[2]):
            grasp_pose = to_pos_quat(grasp)
            qi = self.model.static_ik(grasp_pose, q)
            print("grasp xyz =", grasp_pose[0])

            if qi is not None:
                base_theta_movement = np.abs(q[HelloStretchIdx.BASE_THETA] - qi[HelloStretchIdx.BASE_THETA])
                if base_theta_movement > 0.05:
                    # Prevent large base movements
                    continue
                print(" - IK found")
                print("base_theta_movement", base_theta_movement)
                self.model.set_config(qi)
                input('---')
            else:
                # Grasp attempt failure
                continue
            # Record the initial q value here and use it 
            theta0 = q[2]
            q1 = qi.copy()
            # q1[HelloStretchIdx.LIFT] += 0.08
            q1[HelloStretchIdx.LIFT] += 0.2
            if q1 is not None:
                # Run a validity check to make sure we can actually pick this thing up
                if not self.model.validate(q1):
                    print("invalid standoff config:", q1)
                    continue

                print("found standoff")
                if debug and world_pcd is not None and image_rgb is not None:
                    print("Trying to reach grasp:")
                    print(grasp)
                    fk_pose = self.model.fk(qi, as_matrix=True)

                    # Visualize in RVis
                    for id, grasp in [("executed_grasp", grasp), ("predicted_grasp", orig_grasp), ("fk_pose", fk_pose)]:
                        t = TransformStamped()
                        t.header.stamp = rospy.Time.now()
                        t.child_frame_id = id
                        t.header.frame_id = "map"
                        t.transform = ros_pose_to_transform(matrix_to_pose_msg(grasp))
                        self.grasp_client.broadcaster.sendTransform(t)

                    # Visualize in Open3D
                    show_point_cloud(world_pcd, image_rgb, orig=np.zeros(3), grasps=[grasp, orig_grasp, fk_pose])

                q2 = qi
                # q2 = model.static_ik(grasp_pose, q1)
                if q2 is not None:
                    # if np.abs(eq1) < 0.075 and np.abs(eq2) < 0.075:
                    # go to the grasp and try it
                    q[HelloStretchIdx.LIFT] = 0.99
                    self.manip.goto(q, move_base=False, wait=True, verbose=False)
                    if pause:
                        input('--> go high')
                    q_pre = q.copy()
                    q_pre[5:] = q1[5:]
                    q_pre = self.model.update_gripper(q_pre, open=True)
                    # TODO replace this
                    #self.manip.move_base(theta=q1[2])
                    time.sleep(2.0)
                    self.manip.goto(q_pre, move_base=False, wait=False, verbose=False)
                    self.model.set_config(q1)
                    if pause:
                        input('--> gripper ready; go to standoff')
                    q1 = self.model.update_gripper(q1, open=True)
                    self.manip.goto(q1, move_base=False, wait=True, verbose=False)
                    if pause:
                        input('--> go to grasp')
                    # TODO replace this
                    #self.manip.move_base(theta=q2[2])
                    time.sleep(2.0)
                    self.manip.goto(q_pre, move_base=False, wait=False, verbose=False)
                    self.model.set_config(q2)
                    q2 = self.model.update_gripper(q2, open=True)
                    self.manip.goto(q2, move_base=False, wait=True, verbose=True)
                    if pause:
                        input('--> close the gripper')
                    q2 = self.model.update_gripper(q2, open=False)
                    self.manip.goto(q2, move_base=False, wait=False, verbose=True)
                    time.sleep(2.)
                    q = self.model.update_gripper(q, open=False)
                    self.manip.goto(q, move_base=False, wait=True, verbose=False)
                    # TODO replace this
                    #self.manip.move_base(theta=q[0])
                    return True

        return False

    def pick(self, object: str, debug=True):
        """
        Mobile grasping of an object category present in the last frame.
        
        Arguments:
            object: category of object to grasp
        """
        print(f"Starting pick {object}")
        assert object in ["cup", "bottle"]
        category_id = coco_categories[object]

        # Look at end effector and wait long enough that we have a new observation
        # goal_q = STRETCH_HOME_Q.copy()
        # goal_q[HelloStretchIdx.LIFT] = 1.
        # self.manip.goto(goal_q, wait=False)
        # rospy.sleep(4.)
        self.manip.stow(wait=False)
        rospy.sleep(4.)
        self.manip.look_at_ee()
        rospy.sleep(0.5)
        
        grasp_attempts_made = 0
        for attempt in range(self.num_segment_attempts):
            info = self.slam.get_last_position_vis_info()

            image_object_mask = info["unfiltered_semantic_frame"][:, :, category_id]
            flat_object_mask = image_object_mask.reshape(-1)
            semantic_frame = info["semantic_frame_vis"]
            obstacle_map = info["semantic_map"][0]
            object_map = info["semantic_map"][4 + category_id]
            image_rgb = info['rgb']
            depth = info['depth']

            q, _ = self.manip.update()

            # camera_pose = self.manip.get_pose("camera_color_optical_frame")
            # camera_pose[2, 3] += BASE_HEIGHT
            # camera_pose[0, 3] -= 0.11526719
            camera_pose = self.bot.get_camera_transform().value

            print("CAMERA_POSES")
            print("self.bot.get_camera_transform()")
            print(self.bot.get_camera_transform().value)
            print('self.manip.get_pose("camera_color_optical_frame")')
            print(self.manip.get_pose("camera_color_optical_frame"))
            print('self.manip.get_pose("camera_color_frame")')
            print(self.manip.get_pose("camera_color_frame"))
            print()

            flat_pcd = get_pcd_in_cam(depth, self.intrinsic_mat)

            if attempt == 0:
                print(list(info.keys()))
                print("image_rgb.shape", image_rgb.shape)
                print("flat_pcd.shape", flat_pcd.shape)
                print("image_object_mask.shape", image_object_mask.shape)
                print("obstacle_map.shape", obstacle_map.shape)
                print("object_map.shape", object_map.shape)
                print("intrinsics mat:", self.intrinsic_mat)

                print("Here is how to transform robot coordinates to map coordinates:")

                pose_of_last_map_update = info["pose"]
                pose_in_map_coordinates = self.slam.robot2map(
                    pose_of_last_map_update[:2])

                print("pose_of_last_map_update", pose_of_last_map_update)
                print("curr_pose_in_map_coordinates", pose_in_map_coordinates)
                print()

            num_obj_pts = np.sum(flat_object_mask)
            print(attempt, "Detected this many object points:", num_obj_pts)
            if num_obj_pts < self.min_obj_pts:
                print("Too few object points; trying to segment again...")
                continue

            print("Attempting to grasp...")
            #to_npy_file('stretch2', xyz=flat_pcd, rgb=image_rgb,
            #            depth=depth, xyz_color=image_rgb, seg=flat_object_mask,
            #            K=self.intrinsic_mat)
            predicted_grasps = self.grasp_client.request(flat_pcd,
                                                         image_rgb,
                                                         flat_object_mask,
                                                         frame="camera_color_optical_frame")

            print("options =", [(k, v[-1].shape) for k, v in predicted_grasps.items()])

            # if debug:
            #     cv2.imwrite("semantic_frame.png", semantic_frame)
            #     cv2.imwrite("image_object_mask.png", (image_object_mask * 255).astype(np.uint8))

            predicted_grasps, scores = predicted_grasps[0]
            if len(scores) < self.min_predicted_grasps:
                print("Too few predicted grasps; trying to segment again...")
                continue

            if debug:
                rotated_grasps = [self.R_stretch_camera.T @ grasp for grasp in predicted_grasps]
                show_point_cloud(flat_pcd, image_rgb, orig=np.zeros(3), grasps=rotated_grasps)

            world_grasps = [camera_pose @ grasp for grasp in predicted_grasps]
            world_pcd = trimesh.transform_points(flat_pcd, self.R_stretch_camera)
            world_pcd = trimesh.transform_points(world_pcd, camera_pose)

            if debug:
                show_point_cloud(world_pcd, image_rgb, orig=np.zeros(3), grasps=world_grasps)

            #self.manip.goto_static_grasp(world_grasps, scores, pause=True)
            self.goto_static_grasp(world_grasps, scores, world_pcd, image_rgb, pause=False, debug=debug)
            break

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
                debug=False,
                visualize=True,
                max_steps=400,
                start_with_panorama=False,
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
        # default="192.168.0.49",  # ROBOT 1
        default="192.168.0.48",  # ROBOT 2
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
