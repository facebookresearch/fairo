import a0
import geometry_msgs
import moveit_commander
import moveit_msgs
import os
import json
import rospy
import signal
import subprocess
import typing
from rospy_message_converter import message_converter

os.environ["PYTHONUNBUFFERED"] = "1"


def launch_helper(cmd, ready_substr, verbose=False):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    while True:
        line = proc.stdout.readline()
        if verbose:
            print(line)
        if not line or ready_substr in line:
            break

    return proc


def launch_roscore():
    return launch_helper(["roscore"], b"process[master]: started with pid")


def launch_panda_demo():
    return launch_helper(
        [
            "roslaunch",
            "--wait",
            "moveit_resources_panda_moveit_config",
            "demo.launch",
            "use_rviz:=false",
        ],
        b"You can start planning now!",
    )


launch_roscore()
launch_panda_demo()
rospy.init_node("panda_planner")


def ros2dict(msg):
    return message_converter.convert_ros_message_to_dictionary(msg)


def dict2ros(type_, dict_):
    return message_converter.convert_dictionary_to_ros_message(type_, dict_)


class PandaMoveGroup:
    def __init__(self):
        self._move_group = moveit_commander.MoveGroupCommander("panda_arm")

    def allow_looking(self, value: bool) -> None:
        return self._move_group.allow_looking(value)

    def allow_replanning(self, value: bool) -> None:
        return self._move_group.allow_replanning(value)

    def attach_object(
        self, object_name: str, link_name: str = "", touch_links: typing.List[str] = []
    ) -> bool:
        return self._move_group.attach_object(object_name, link_name, touch_links)

    def clear_path_constraints(self) -> None:
        return self._move_group.clear_path_constraints()

    def clear_pose_target(self, end_effector_link: str = "") -> None:
        return self._move_group.detach_object(end_effector_link)

    def clear_pose_targets(self) -> None:
        return self._move_group.clear_pose_targets()

    def clear_trajectory_constraints(self) -> None:
        return self._move_group.clear_trajectory_constraints()

    def compute_cartesian_path(
        self,
        waypoints: typing.List["ros2dict[geometry_msgs.Pose]"],
        eef_step: float,
        jump_threshold: float,
        avoid_collisions: bool = True,
        path_constraints: "ros2dict[moveit_msgs.Constraints]" = None,
    ) -> float:
        waypoints = [dict2ros("geometry_msgs/Pose", waypoint) for waypoint in waypoints]
        if path_constraints:
            path_constraints = dict2ros("moveit_msgs/Constraints", path_constraints)
        return self._move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions, path_constraints
        )

    def construct_motion_plan_request(self) -> "ros2dict[moveit_msgs.MotionPlanRequest]":
        return ros2dict(self._move_group.construct_motion_plan_request())

    def detach_object(self, name: str = "") -> bool:
        return self._move_group.detach_object(name)

    def forget_joint_values(self, name: str) -> None:
        return self._move_group.forget_joint_values(name)

    def get_active_joints(self) -> typing.List[str]:
        return self._move_group.get_active_joints()

    def get_current_joint_values(self) -> typing.List[float]:
        return self._move_group.get_current_joint_values()

    def get_current_pose(self, end_effector_link: str = "") -> "ros2dict[geometry_msgs.Pose]":
        return ros2dict(self._move_group.get_current_pose(end_effector_link))["pose"]

    def get_current_rpy(self, end_effector_link: str = "") -> typing.List[float]:
        return self._move_group.get_current_rpy(end_effector_link)

    def get_current_state(self) -> "ros2dict[moveit_msgs.RobotState]":
        return ros2dict(self._move_group.get_current_state())

    def get_end_effector_link(self) -> str:
        return self._move_group.get_end_effector_link()

    def get_goal_tolerance(self) -> typing.List[float]:
        return self._move_group.get_goal_tolerance()

    def get_goal_joint_tolerance(self) -> float:
        return self._move_group.get_goal_joint_tolerance()

    def get_goal_orientation_tolerance(self) -> float:
        return self._move_group.get_goal_orientation_tolerance()

    def get_goal_position_tolerance(self) -> float:
        return self._move_group.get_goal_position_tolerance()

    def get_interface_description(self) -> "ros2dict[moveit_msgs.PlannerInterfaceDescription]":
        return ros2dict(self._move_group.get_interface_description())

    def get_joint_value_target(self) -> typing.List[float]:
        return self._move_group.get_joint_value_target()

    def get_joints(self) -> typing.List[str]:
        return self._move_group.get_joints()

    def get_known_constraints(self) -> typing.List[str]:
        return self._move_group.get_known_constraints()

    def get_name(self) -> str:
        return self._move_group.get_name()

    def get_named_target_values(self, target: str) -> typing.Dict[str, float]:
        return self._move_group.get_named_target_values(target)

    def get_named_targets(self) -> typing.List[str]:
        return self._move_group.get_named_targets()

    def get_path_constraints(self) -> "ros2dict[moveit_msgs.Constraints]":
        return ros2dict(self._move_group.get_path_constraints())

    def get_planner_id(self) -> str:
        return self._move_group.get_planner_id()

    def get_planning_frame(self) -> str:
        return self._move_group.get_planning_frame()

    def get_planning_time(self) -> float:
        return self._move_group.get_planning_time()

    def get_pose_reference_frame(self) -> str:
        return self._move_group.get_pose_reference_frame()

    def get_random_joint_values(self) -> typing.List[float]:
        return self._move_group.get_random_joint_values()

    def get_random_pose(self, end_effector_link: str = "") -> "ros2dict[geometry_msgs.Pose]":
        return ros2dict(self._move_group.get_random_pose(end_effector_link))["pose"]

    def get_remembered_joint_values(self) -> typing.Dict[str, typing.List[float]]:
        return self._move_group.get_remembered_joint_values()

    def get_trajectory_constraints(self) -> "ros2dict[moveit_msgs.TrajectoryConstraints]":
        return ros2dict(self._move_group.get_trajectory_constraints())

    def get_variable_count(self) -> int:
        return self._move_group.get_variable_count()

    def has_end_effector_link(self) -> bool:
        return self._move_group.has_end_effector_link()

    def has_end_effector_link(self, link_name: str) -> bool:
        return self._move_group.has_end_effector_link(link_name)

    def pick(
        self,
        object_name: str,
        grasps: typing.List["ros2dict[moveit_msgs.Grasp]"] = [],
        plan_only: bool = False,
    ) -> int:
        grasps = [dict2ros("moveit_msgs/Grasp", grasp) for grasp in grasps]
        return self._move_group.pick(object_name, grasps, plan_only)

    def place(
        self,
        object_name: str,
        locations: typing.List["ros2dict[moveit_msgs.PlaceLocation]"] = [],
        plan_only: bool = False,
    ) -> int:
        locations = [dict2ros("moveit_msgs/PlaceLocation", location) for location in locations]
        return self._move_group.place(object_name, locations, plan_only)

    def plan(
        self,
    ) -> typing.Tuple[
        bool,
        "ros2dict[moveit_msgs.TrajectoryConstraints]",
        float,
        "ros2dict[moveit_msgs.MoveItErrorCodes]",
    ]:
        success, trajectory_msg, planning_time, error_code = self._move_group.plan()
        trajectory_msg = ros2dict(trajectory_msg)
        error_code = ros2dict(error_code)
        return success, trajectory_msg, planning_time, error_code

    def remember_joint_values(self, name: str, values: typing.List[float] = None) -> None:
        return self._move_group.remember_joint_values(name, values)

    def set_end_effector_link(self, link_name: str) -> bool:
        return self._move_group.set_end_effector_link(link_name)

    def set_goal_joint_tolerance(self, value: float) -> None:
        return self._move_group.set_goal_joint_tolerance(value)

    def set_goal_orientation_tolerance(self, value: float) -> None:
        return self._move_group.set_goal_orientation_tolerance(value)

    def set_goal_position_tolerance(self, value: float) -> None:
        return self._move_group.set_goal_position_tolerance(value)

    def set_goal_tolerance(self, value: float) -> None:
        return self._move_group.set_goal_tolerance(value)

    def set_max_acceleration_scaling_factor(self, value: float) -> None:
        return self._move_group.set_max_acceleration_scaling_factor(value)

    def set_max_velocity_scaling_factor(self, value: float) -> None:
        return self._move_group.set_max_velocity_scaling_factor(value)

    def set_named_target(self, name: str) -> bool:
        return self._move_group.set_named_target(name)

    def set_num_planning_attempts(self, num_planning_attempts: int) -> None:
        return self._move_group.set_num_planning_attempts(num_planning_attempts)

    def set_planner_id(self, planner_id: str) -> None:
        return self._move_group.set_planner_id(planner_id)

    def set_planning_pipeline_id(self, planning_pipeline: str) -> None:
        return self._move_group.set_planning_pipeline_id(planning_pipeline)

    def set_planning_time(self, seconds: float) -> None:
        return self._move_group.set_planning_time(seconds)

    def set_pose_reference_frame(self, reference_frame: str) -> None:
        return self._move_group.set_pose_reference_frame(reference_frame)

    def set_pose_target(
        self, pose: "ros2dict[geometry_msgs.Pose]", end_effector_link: str = ""
    ) -> bool:
        pose = dict2ros("geometry_msgs/Pose", pose)
        return self._move_group.set_pose_target(pose, end_effector_link)

    def set_pose_targets(
        self, poses: "ros2dict[geometry_msgs.Pose]", end_effector_link: str = ""
    ) -> bool:
        poses = [dict2ros("geometry_msgs/Pose", pose) for pose in poses]
        return self._move_group.set_pose_targets(poses, end_effector_link)

    def set_position_target(
        self, xyz: typing.Tuple[float, float, float], end_effector_link: str = ""
    ) -> bool:
        return self._move_group.set_position_target(xyz, end_effector_link)

    def set_rpy_target(
        self, rpy: typing.Tuple[float, float, float], end_effector_link: str = ""
    ) -> bool:
        return self._move_group.set_rpy_target(rpy, end_effector_link)

    def set_random_target(self) -> None:
        return self._move_group.set_random_target()

    def set_start_state(self, msg: "ros2dict[moveit_msgs.RobotState]") -> None:
        msg = dict2ros("moveit_msgs/RobotState", msg)
        return self._move_group.set_start_state(msg)

    def set_start_state_to_current_state(self) -> None:
        return self._move_group.set_start_state_to_current_state()

    def set_support_surface_name(self, value: str) -> None:
        return self._move_group.set_support_surface_name(value)

    def set_trajectory_constraints(
        self, value: "ros2dict[moveit_msgs.TrajectoryConstraints]"
    ) -> None:
        value = dict2ros("moveit_msgs/TrajectoryConstraints", value)
        return self._move_group.set_trajectory_constraints(value)

    def stop(self) -> None:
        return self._move_group.stop()


class PlanningScene:
    def __init__(self):
        self._scene = moveit_commander.PlanningSceneInterface()

    def add_box(
        self,
        name: str,
        pose: "ros2dict[geometry_msgs.Pose]",
        size: typing.Tuple[float, float, float] = (1, 1, 1),
    ) -> None:
        pose_stamped = geometry_msgs.msg.PoseStamped(pose=dict2ros("geometry_msgs/Pose", pose))
        return self._scene.add_box(name, pose_stamped, size)

    def add_cylinder(
        self, name: str, pose: "ros2dict[geometry_msgs.Pose]", height: float, radius: float
    ) -> None:
        pose_stamped = geometry_msgs.msg.PoseStamped(pose=dict2ros("geometry_msgs/Pose", pose))
        return self._scene.add_cylinder(name, pose_stamped, height, radius)

    # def add_mesh()
    # def add_object()

    def add_plane(
        self,
        name: str,
        pose: "ros2dict[geometry_msgs.Pose]",
        normal: typing.Tuple[float, float, float] = (0, 0, 1),
        offset: float = 0,
    ) -> None:
        pose_stamped = geometry_msgs.msg.PoseStamped(pose=dict2ros("geometry_msgs/Pose", pose))
        return self._scene.add_plane(name, pose_stamped, normal, offset)

    def add_sphere(
        self, name: str, pose: "ros2dict[geometry_msgs.Pose]", radius: float = 1
    ) -> None:
        pose_stamped = geometry_msgs.msg.PoseStamped(pose=dict2ros("geometry_msgs/Pose", pose))
        return self._scene.add_sphere(name, pose_stamped, radius)

    # def apply_planning_scene()

    def attach_box(
        self,
        link: str,
        name: str,
        pose: "ros2dict[geometry_msgs.Pose]" = None,
        size: typing.Tuple[float, float, float] = (1, 1, 1),
    ):
        pose_stamped = geometry_msgs.msg.PoseStamped(pose=dict2ros("geometry_msgs/Pose", pose))
        return self._scene.attach_box(link, name, pose_stamped, size)

    # def attach_mesh()
    # def attach_object()

    def clear(self) -> None:
        return self._scene.clear()

    # def get_attached_objects()
    # def get_known_object_names()
    # def get_known_object_names_in_roi()
    # def get_object_poses()
    # def get_objects()
    # def remove_attached_object()
    # def remove_world_object()


move_group = PandaMoveGroup()
scene = PlanningScene()


def onrequest(req):
    pkt = req.pkt
    print(f"Got request: {pkt.payload}")
    try:
        req.reply(json.dumps({"err": "success", "result": eval(pkt.payload.decode())}))
        print("Done.")
    except Exception as e:
        req.reply(json.dumps({"err": "failed", "result": str(e)}))
        print("Failed.")


server = a0.RpcServer("panda_planner", onrequest, None)
signal.pause()
