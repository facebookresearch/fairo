"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import time
import math
import logging
import numpy as np
import os
import math
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode
from droidlet.task.task import Task, BaseMovementTask
from droidlet.memory.memory_nodes import TaskNode, TripleNode
from droidlet.lowlevel.robot_coordinate_utils import base_canonical_coords_to_pyrobot_coords
from droidlet.interpreter.robot.default_behaviors import get_distant_goal
from droidlet.interpreter.robot.objects import DanceMovement

from droidlet.lowlevel.robot_mover_utils import (
    get_move_target_for_point,
    ARM_HEIGHT,
    get_camera_angles,
    TrajectoryDataSaver,
    visualize_examine,
    get_step_target_for_straightline_move,
    CAMERA_HEIGHT,
    get_circular_path,
)


# TODO make this highest priority, hardstop, etc
class Stop(Task):
    def __init__(self, agent, task_data, featurizer=None):
        super().__init__(agent)
        TaskNode(self.agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        self.agent.mover.stop()
        self.finished = True


# FIXME store dances, etc.
class Dance(Task):
    def __init__(self, agent, task_data, featurizer=None):
        super().__init__(agent)
        # movement should be a Movement object from dance.py
        self.movement = DanceMovement(self.agent, None)
        self.movement_type = task_data.get("movement_type", None)
        TaskNode(self.agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        self.interrupted = False

        if self.movement_type == "wave":
            self.movement.wave()

        elif not self.movement:  # default move
            mv = Move(self.agent, {"target": [-1000, -1000, -1000], "approx": 2})
            self.add_child_task(mv)

        self.finished = True


def torad(deg):
    if deg:
        return deg * math.pi / 180


#### TODO, FIXME!:
#### merge Look, Point, Turn into dancemove; on mc side too
class Look(Task):
    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.task_data = task_data
        assert (
            task_data.get("target")
            or task_data.get("pitch")
            or task_data.get("yaw")
            or task_data.get("relative_yaw")
            or task_data.get("relative_pitch")
        )
        self.command_sent = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        self.finished = False
        self.interrupted = False
        if not self.command_sent:
            if self.task_data.get("target"):
                status = self.agent.mover.look_at(self.task_data["target"])
            if self.task_data.get("pitch") or self.task_data.get("yaw"):
                status = self.agent.mover.set_look(
                    torad(self.task_data.get("yaw")), torad(self.task_data.get("pitch"))
                )
            if self.task_data.get("relative_pitch") or self.task_data.get("relative_yaw"):
                status = self.agent.mover.relative_pan_tilt(
                    torad(self.task_data.get("relative_yaw")),
                    torad(self.task_data.get("relative_pitch")),
                )
            self.command_sent = True
            if status == "finished":
                self.finished = True
        else:
            self.finished = self.agent.mover.bot_step()

    def __repr__(self):
        if self.task_data.get("target"):
            target = self.task_data.get["target"]
            return "<Look at {} {} {}>".format(target[0], target[1], target[2])
        else:
            return "<Look: {}".format(self.task_data)


class Point(Task):
    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.target = np.asarray(task_data["target"])
        print(f"type {type(self.target), self.target}")
        self.steps = ["not_started"] * 2
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def get_pt_from_region(self, region):
        assert (
            len(region) == 6
        ), "Region list has less than 6 elements (minx, miny, minz, maxx, maxy, maxz)"
        return region[:3]  # just return the min xyz for now

    @Task.step_wrapper
    def step(self):
        self.interrupted = False
        print(f"target {self.target}")
        pt = self.get_pt_from_region(self.target.tolist())
        logging.info(f"Calling bot to Point at {pt}")
        logging.info(f"base pos {self.agent.mover.get_base_pos_in_canonical_coords()}")

        # Step 1 - Move close to the object.
        if self.steps[0] == "not_started":
            base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
            target = get_move_target_for_point(base_pos, pt)
            logging.info(f"Move Target for point {target}")
            self.add_child_task(Move(self.agent, {"target": target}))
            self.steps[0] = "finished"
            return

        # Step 2 - Turn so that the object is in FOV
        if self.steps[0] == "finished" and self.steps[1] == "not_started":
            base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
            yaw_rad, _ = get_camera_angles([base_pos[0], ARM_HEIGHT, base_pos[1]], pt)
            self.add_child_task(Turn(self.agent, {"yaw": yaw_rad}))
            self.steps[1] = "finished"
            return

        # Step 3 - Point at the object
        if self.steps[0] == "finished" and self.steps[1] == "finished":
            status = self.agent.mover.point_at(pt)
            if status == "finished":
                self.finished = True

    def __repr__(self):
        return "<Point at {}>".format(self.target)


class Move(BaseMovementTask):
    def __init__(self, agent, task_data, featurizer=None):
        super().__init__(agent, task_data)
        self.target = np.array(task_data["target"])
        self.is_relative = task_data.get("is_relative", 0)
        self.path = None
        self.command_sent = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def target_to_memory(self, target):
        return [target[0], 0, target[1]]

    @Task.step_wrapper
    def step(self):
        self.interrupted = False
        self.finished = False
        if not self.command_sent:
            logging.info("calling move with : %r" % (self.target.tolist()))
            self.command_sent = True
            if self.is_relative:
                self.agent.mover.move_relative([self.target.tolist()], blocking=False)
            else:
                self.agent.mover.move_absolute([self.target.tolist()], blocking=False)

        else:
            self.finished = not self.agent.mover.is_busy()

    def __repr__(self):
        return "<Move {}>".format(self.target)


class Turn(Task):
    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.yaw = task_data.get("yaw") or task_data.get("relative_yaw")
        self.command_sent = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        self.interrupted = False
        self.finished = False
        if not self.command_sent:
            self.command_sent = True
            self.agent.mover.turn(self.yaw)
        else:
            self.finished = not self.agent.mover.is_busy()

    def __repr__(self):
        return "<Turn {} degrees>".format(self.yaw)


# TODO handle case where agent already has item in inventory (pure give)
class Get(Task):
    def __init__(self, agent, task_data):
        super().__init__(agent)
        # get target should be a ReferenceObjectNode memid
        self.get_target = task_data["get_target"]
        self.give_target = task_data["give_target"]
        # steps take values "not_started", "started", "complete"
        if not self.give_target:
            # TODO all movements simultaneous- change look while driving
            # approach_pickup, look_at_object, grab
            self.steps = ["not_started"] * 3
        else:
            # approach_pickup, look_at_object, grab, approach_dropoff, give/drop
            self.steps = ["not_started"] * 5
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def get_mv_target(self, get_or_give="get", end_distance=0.35):
        """figure out the location where agent should move to in order to get or give object in global frame
        all units are in metric unit

        Args:
            get_or_give (str, optional): whether to get or give object. Defaults to "get".
            end_distance (float, optional): stand end_distance away from the goal in meter. Defaults to 0.35.

        Returns:
            [tuple]: (x,y,theta) location the agent should move to, in global co-ordinate system
        """
        agent_pos = np.array(self.agent.mover.get_base_pos())[:2]
        if get_or_give == "get":
            target_memid = self.get_target
        else:
            target_memid = self.give_target
        target_pos = self.agent.memory.get_mem_by_id(target_memid).get_pos()
        target_pos = np.array((target_pos[0], target_pos[2]))
        diff = target_pos - agent_pos
        distance = np.linalg.norm(diff)
        # FIXME add a check to make sure not already there
        xz = agent_pos + (distance - end_distance) * diff / distance
        # TODO: Check if yaw s right
        target_yaw = np.arctan2(diff[1], diff[0])
        received_yaw = False
        while not received_yaw:
            try:
                target_yaw += self.agent.mover.get_base_pos()[2]
                received_yaw = True
            except:
                time.sleep(0.1)
        return (xz[0], xz[1], target_yaw)

    @Task.step_wrapper
    def step(self):
        agent = self.agent
        self.interrupted = False
        self.finished = False
        # move to object to be picked up
        if self.steps[0] == "not_started":
            # check if already holding target object for pure give, when object is grasped
            # its added to memory with tag "_in_inventory"
            if self.get_target in agent.memory.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(
                agent.memory, "_in_inventory"
            ):
                self.steps[0] = "finished"
                self.steps[1] = "finished"
                self.steps[2] = "finished"
            else:
                target = self.get_mv_target(get_or_give="get")
                self.add_child_task(Move(agent, {"target": target}))
                # TODO a loop?  otherwise check location/graspability instead of just assuming?
                self.steps[0] = "finished"
            return
        # look at the object directly
        if self.steps[0] == "finished" and self.steps[1] == "not_started":
            target_pos = agent.memory.get_mem_by_id(self.get_target).get_pos()
            self.add_child_task(Look(agent, {"target": target_pos}))
            self.steps[1] = "finished"
            return
        # grab it
        if self.steps[1] == "finished" and self.steps[2] == "not_started":
            self.add_child_task(AutoGrasp(agent, {"target": self.get_target}))
            self.steps[2] = "finished"
            return
        if len(self.steps) == 3:
            self.finished = True
            return
        # go to the place where you are supposed to drop off the item
        if self.steps[3] == "not_started":
            target = self.get_mv_target(agent, get_or_give="give")
            self.add_child_task(Move(agent, {"target": target}))
            # TODO a loop?  otherwise check location/graspability instead of just assuming?
            self.steps[3] = "finished"
            return
        # drop it
        if self.steps[3] == "finished":
            self.add_child_task(Drop(agent, {"object": self.get_target}))
            self.finished = True
            return

    def __repr__(self):
        return "<get {}>".format(self.get_target)


class AutoGrasp(Task):
    """thin wrapper for Dhiraj' grasping routine."""

    def __init__(self, agent, task_data):
        super().__init__(agent)
        # this is a ref object memid
        self.target = task_data["target"]
        self.command_sent = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        self.interrupted = False
        self.finished = False
        if not self.command_sent:
            self.command_sent = True
            self.agent.mover.grab_nearby_object()
        else:
            self.finished = self.agent.mover.bot_step()
            # TODO check that the object in the gripper is actually the object we meant to pick up
            # TODO deal with failure cases
            # TODO tag this in the grip task, not here
            if self.finished:
                if self.agent.mover.is_object_in_gripper():
                    self.agent.memory.nodes[TripleNode.NODE_TYPE].tag(
                        self.agent.memory, self.target, "_in_inventory"
                    )


class Drop(Task):
    """drop whatever is in hand."""

    def __init__(self, agent, task_data):
        super().__init__(agent)
        # currently unused, we can expand this soon?
        self.object_to_drop = task_data.get("object", None)
        self.command_sent = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        agent = self.agent
        self.interrupted = False
        self.finished = False
        if not self.command_sent:
            logging.info("Dropping the object in hand")
            self.command_sent = True
            agent.mover.drop()
        else:
            self.finished = agent.mover.bot_step() and not agent.mover.is_object_in_gripper()
            if self.finished:
                agent.memory.nodes[TripleNode.NODE_TYPE].untag(
                    agent.memory, self.object_to_drop, "_in_inventory"
                )
                if self.object_to_drop is None:
                    # assumed there is only one object with tag "_in_inventory"
                    for mmid in agent.memory.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(
                        agent.memory, "_in_inventory"
                    ):
                        agent.memory.nodes[TripleNode.NODE_TYPE].untag(
                            agent.memory, mmid, "_in_inventory"
                        )


class TrajectorySaverTask(Task):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.save_data = task_data.get("save_data", False)
        self.data_path = task_data.get("data_path", "default")
        self.dbg_str = "None"
        if self.save_data:
            self.data_saver = TrajectoryDataSaver(
                os.path.join(agent.opts.data_store_path, self.data_path)
            )
            self.data_savers = [self.data_saver]
            parent_data_saver = task_data.get("parent_data_saver", None)
            if parent_data_saver is not None:
                self.data_savers.append(parent_data_saver)
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def save_rgb_depth_seg(self):
        rgb, depth, segm = self.agent.mover.get_rgb_depth_segm()
        # store depth in mm
        depth *= 1e3
        depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
        depth = depth.astype(np.uint16)

        pos = self.agent.mover.get_base_pos()
        habitat_pos, habitat_rot = self.agent.mover.bot.get_habitat_state()
        for data_saver in self.data_savers:
            data_saver.set_dbg_str(self.dbg_str)
            data_saver.save(rgb, depth, segm, pos, habitat_pos, habitat_rot)

    @Task.step_wrapper
    def step(self):
        if self.save_data:
            self.save_rgb_depth_seg()

    def __repr__(self):
        return "<TrajectorySaverTask>"


class CuriousExplore(TrajectorySaverTask):
    """use slam to explore environemt, but also examine detections"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.steps = ["not_started"] * 2
        self.task_data = task_data
        self.goal = task_data.get("goal", (19, 19, 0))
        self.init_curious_logger()
        self.agent = agent
        self.objects_examined = 0
        self.save_data = task_data.get("save_data")
        print(f"CuriousExplore task_data {task_data}")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def init_curious_logger(self):
        self.logger = logging.getLogger("curious")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            f"curious_explore_{'_'.join([str(x) for x in self.goal])}.log", "w"
        )
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(filename)s:%(lineno)s - %(funcName)s(): %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(f"CuriousExplore task_data {self.task_data}")

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False

        ExaminedMap = self.agent.memory.place_field

        if self.steps[0] == "not_started":
            self.logger.info(f"exploring goal {self.goal}")
            self.agent.mover.explore(self.goal)
            self.dbg_str = "Explore"
            self.steps[0] = "finished"
            return

        # execute a examine maneuver
        if self.steps[0] == "finished" and self.steps[1] == "not_started":
            # Get a list of current detections
            objects = DetectedObjectNode.get_all(self.agent.memory)
            pos = self.agent.mover.get_base_pos_in_canonical_coords()

            # pick all from unexamined, in-sight object
            def pick_random_in_sight(objects, base_pos):
                for x in objects:
                    if ExaminedMap.can_examine(x):
                        # check for line of sight and if within a certain distance
                        yaw_rad, _ = get_camera_angles(
                            [base_pos[0], CAMERA_HEIGHT, base_pos[1]], x["xyz"]
                        )
                        dist = np.linalg.norm(base_pos[:2] - [x["xyz"][0], x["xyz"][2]])
                        if abs(yaw_rad - base_pos[2]) <= math.pi / 4 and dist <= 3:
                            self.logger.info(
                                f"Exploring eid {x['eid']}, {x['label']} next, dist {dist}"
                            )
                            return x
                return None

            target = pick_random_in_sight(objects, pos)
            if target is not None:
                self.logger.info(
                    f"CuriousExplore Target {target['eid'], target['label'], target['xyz']}, robot pos {pos}"
                )
                ExaminedMap.update(target)
                self.dbg_str = f"Examine {str(target['eid']) + '_' + str(target['label'])} xyz {str(np.round(target['xyz'],3))}"
                if os.getenv("HEURISTIC") == "straightline":
                    examine_heuristic = ExamineDetectionStraightline
                else:
                    examine_heuristic = ExamineDetectionCircle
                self.add_child_task(
                    examine_heuristic(
                        self.agent,
                        {
                            "target": target,
                            "save_data": self.save_data,
                            "vis_path": f"{self.task_data['data_path']}",
                            "data_path": f"{self.task_data['data_path']}/{str(self.objects_examined)}",
                            "dbg_str": self.dbg_str,
                            "parent_data_saver": self.data_savers[0],
                            "logger": self.logger,
                        },
                    )
                )
                self.objects_examined += 1
            self.steps[1] = "finished"
            return

        else:
            self.finished = self.agent.mover.nav.is_done_exploring().value
            if not self.finished:
                self.steps = ["not_started"] * 2
            else:
                self.logger.info("Exploration finished!")

    def __repr__(self):
        return "<CuriousExplore>"


class ExamineDetectionStraightline(TrajectorySaverTask):
    """Examine a detection"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.task_data = task_data
        self.target = task_data["target"]
        self.frontier_center = np.asarray(self.target["xyz"])
        self.agent = agent
        self.last_base_pos = None
        self.robot_poses = []
        self.dbg_str = task_data.get("dbg_str")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False
        logger = logging.getLogger("curious")
        base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
        self.robot_poses.append(base_pos)
        dist = np.linalg.norm(
            base_pos[:2] - np.asarray([self.frontier_center[0], self.frontier_center[2]])
        )
        logger.info(f"Deciding examination, dist = {dist}")
        d = 1
        if self.last_base_pos is not None:
            d = np.linalg.norm(base_pos[:2] - self.last_base_pos[:2])
            # logger.info(f"Distance moved {d}")
        if (base_pos != self.last_base_pos).any() and dist > 0.2 and d > 0:
            tloc = get_step_target_for_straightline_move(base_pos, self.frontier_center)
            logger.debug(
                f"get_step_target_for_straight_move \
                \nx, z, yaw = {base_pos},\
                \nxf, zf = {self.frontier_center[0], self.frontier_center[2]} \
                \nx_move, z_move, yaw_move = {tloc}"
            )
            logging.info(f"Current Pos {base_pos}")
            logging.info(f"Move Target for Examining {tloc}")
            logging.info(f"Distance being moved {np.linalg.norm(base_pos[:2]-tloc[:2])}")
            self.add_child_task(Move(self.agent, {"target": tloc}))

            # visualize tloc, frontier_center, obstacle_map
            if os.getenv("VISUALIZE_EXAMINE", "False") == "True":
                visualize_examine(
                    self.agent,
                    self.robot_poses,
                    self.frontier_center,
                    self.target["label"],
                    self.agent.mover.get_obstacles_in_canonical_coords(),
                    self.task_data["vis_path"],
                )

            self.last_base_pos = base_pos
            return
        else:
            logger.info("Finished Examination")
            self.finished = self.agent.mover.bot_step()

    def __repr__(self):
        return "<ExamineDetectionStraightline {}>".format(self.target["label"])


class ExamineDetectionCircle(TrajectorySaverTask):
    """Examine a detection"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.task_data = task_data
        self.target = task_data["target"]
        self.radius = task_data.get("radius", 0.7)
        self.frontier_center = np.asarray(self.target["xyz"])
        self.agent = agent
        self.steps = 0
        self.robot_poses = []
        self.last_base_pos = None
        self.dbg_str = task_data.get("dbg_str")
        self.logger = task_data.get("logger")
        base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
        self.pts = get_circular_path(
            self.frontier_center, base_pos, radius=self.radius, num_points=40
        )
        self.logger.info(f"{len(self.pts)} pts on cicle {self.pts}")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False
        base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
        if self.steps > 0:  # without any steps, the robot isn't on the circle of inspection
            self.robot_poses.append(base_pos)
        d = 1
        if self.last_base_pos is not None:
            d = np.linalg.norm(base_pos[:2] - self.last_base_pos[:2])
            self.logger.info(f"Distance moved {d}")
        if (base_pos != self.last_base_pos).any() and self.steps < len(self.pts):
            tloc = self.pts[self.steps]
            self.steps += 1
            self.logger.info(f"step {self.steps} moving to {tloc} Current Pos {base_pos}")
            self.add_child_task(Move(self.agent, {"target": tloc}))
            self.last_base_pos = base_pos
            # visualize tloc, frontier_center, obstacle_map
            if os.getenv("VISUALIZE_EXAMINE", "False") == "True":
                visualize_examine(
                    self.agent,
                    self.robot_poses,
                    self.frontier_center,
                    self.target["label"],
                    self.agent.mover.get_obstacles_in_canonical_coords(),
                    self.task_data["vis_path"],
                    self.pts,
                )

            return
        else:
            self.logger.info("Finished Examination")
            self.finished = self.agent.mover.bot_step()

    def __repr__(self):
        return "<ExamineDetectionCircle {}>".format(self.target["label"])


class Explore(TrajectorySaverTask):
    """use slam to explore environemt"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.command_sent = False
        self.agent = agent
        self.goal = task_data.get("goal")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False
        if not self.finished:
            self.agent.mover.explore(self.goal)
            self.finished = self.agent.mover.nav.is_done_exploring().value


class TimedExplore(TrajectorySaverTask):
    """use slam to explore environemt"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.command_sent = False
        self.agent = agent
        self.goal = task_data.get("goal")
        self.timeout = task_data.get("timeout")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False
        if not self.finished and self.timeout > 0:
            self.agent.mover.explore(self.goal)
            self.timeout -= 1
            self.finished = self.agent.mover.is_done_exploring()
        else:
            self.finished = True


class Reexplore(Task):
    """use slam to explore environemt, but also examine detections"""

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.tasks = ["straight", "circle", "circle_big", "random1", "random2"]
        self.task_data = task_data
        self.target = task_data.get("target")
        self.spawn_pos = task_data.get("spawn_pos")
        self.base_pos = task_data.get("base_pos")
        self.init_logger()
        self.agent = agent
        print(f"Reexplore task_data {task_data}")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def init_logger(self):
        logger = logging.getLogger("reexplore")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("reexplore.log", "w")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(filename)s:%(lineno)s - %(funcName)s(): %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f"Reexplore task_data {self.task_data}")

    @Task.step_wrapper
    def step(self):
        super().step()
        self.interrupted = False
        self.finished = False
        logger = logging.getLogger("reexplore")
        task_name = self.tasks.pop(0)

        # execute a straigtline examine
        if task_name == "straight":
            self.agent.mover.bot.respawn_agent(
                self.spawn_pos["position"], self.spawn_pos["rotation"]
            )
            base_pos = self.agent.mover.get_base_pos()
            assert np.allclose(base_pos, self.base_pos)  # checking that poses match

            self.add_child_task(
                ExamineDetectionStraightline(
                    self.agent,
                    {
                        "target": self.target,
                        "save_data": True,
                        "vis_path": f"{self.task_data['data_path']}/s1",
                        "data_path": f"{self.task_data['data_path']}/s1",
                        "dbg_str": f"Straightline examine {self.target}",
                        "logger": logger,
                    },
                )
            )
            return

        # execute a circle examine
        if task_name == "circle":
            self.agent.mover.bot.respawn_agent(
                self.spawn_pos["position"], self.spawn_pos["rotation"]
            )
            base_pos = self.agent.mover.get_base_pos()
            assert np.allclose(base_pos, self.base_pos)

            self.add_child_task(
                ExamineDetectionCircle(
                    self.agent,
                    {
                        "target": self.target,
                        "save_data": True,
                        "vis_path": f"{self.task_data['data_path']}/c1s",
                        "data_path": f"{self.task_data['data_path']}/c1s",
                        "dbg_str": f"Circle examine {self.target}",
                        "logger": logger,
                    },
                )
            )

            return

        # execute a circle examine with radius as distance from spawn loc
        if task_name == "circle_big":
            self.agent.mover.bot.respawn_agent(
                self.spawn_pos["position"], self.spawn_pos["rotation"]
            )
            base_pos = self.agent.mover.get_base_pos()
            assert np.allclose(base_pos, self.base_pos)

            base_pos_can = self.agent.mover.get_base_pos_in_canonical_coords()
            dist = np.linalg.norm(
                base_pos_can[:2] - [self.target["xyz"][0], self.target["xyz"][2]]
            )

            self.add_child_task(
                ExamineDetectionCircle(
                    self.agent,
                    {
                        "target": self.target,
                        "save_data": True,
                        "vis_path": f"{self.task_data['data_path']}/c1l",
                        "data_path": f"{self.task_data['data_path']}/c1l",
                        "dbg_str": f"Circle examine {self.target}",
                        "logger": logger,
                        "radius": dist,
                    },
                )
            )
            return

        if task_name == "random1":
            self.agent.mover.bot.respawn_agent(
                self.spawn_pos["position"], self.spawn_pos["rotation"]
            )
            base_pos = self.agent.mover.get_base_pos()
            assert np.allclose(base_pos, self.base_pos)

            self.add_child_task(
                TimedExplore(
                    self.agent,
                    {
                        "goal": get_distant_goal(base_pos[0], base_pos[1], base_pos[2]),
                        "save_data": os.getenv("SAVE_EXPLORATION", "False") == "True",
                        "data_path": f"{self.task_data['data_path']}/r1",
                        "timeout": 20,
                    },
                )
            )
            return

        if task_name == "random2":
            self.agent.mover.bot.respawn_agent(
                self.spawn_pos["position"], self.spawn_pos["rotation"]
            )
            base_pos = self.agent.mover.get_base_pos()
            assert np.allclose(base_pos, self.base_pos)

            self.add_child_task(
                TimedExplore(
                    self.agent,
                    {
                        "goal": get_distant_goal(base_pos[0], base_pos[1], base_pos[2]),
                        "save_data": os.getenv("SAVE_EXPLORATION", "False") == "True",
                        "data_path": f"{self.task_data['data_path']}/r2",
                        "timeout": 20,
                    },
                )
            )
            return

        else:
            self.finished = True

    def __repr__(self):
        return "<ReExplore>"
