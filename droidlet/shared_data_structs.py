import numpy as np
import time
import open3d as o3d
from droidlet.interpreter.condition import AlwaysCondition, NeverCondition, NotCondition, TaskStatusCondition
from droidlet.lowlevel.locobot.locobot_mover_utils import xyz_pyrobot_to_canonical_coords
from droidlet.memory.memory_nodes import TaskNode, TripleNode
from droidlet.base_util import TICKS_PER_SEC


class Task(object):
    """This class represents a Task, the exact implementation of which
    will depend on the framework and environment. A task can be placed on a
    task stack, and represents a unit (which in itself can contain a sequence of s
    smaller subtasks).

    Attributes:
        memid (string): Memory id of the task in agent's memory
        interrupted (bool): A flag indicating whetherr the task has been interrupted
        finished (bool): A flag indicating whether the task finished
        name (string): Name of the task
        undone (bool): A flag indicating whether the task was undone / reverted
        last_stepped_time (int): Timestamp of last step through the task
        stop_condition (Condition): The condition on which the task will be stopped (by default,
                        this is NeverCondition)
    Examples::
        >>> Task()
    """

    def __init__(self, agent, task_data={}):
        self.agent = agent
        self.run_count = 0
        self.interrupted = False
        self.finished = False
        self.name = None
        self.undone = False
        self.last_stepped_time = None
        self.prio = -1
        self.running = 0
        self.memid = TaskNode.create(self.agent.memory, self)
        # TODO put these in memory in a new table?
        # TODO methods for safely changing these
        i, s, ru, re = self.get_default_conditions(task_data, agent)
        self.init_condition = i
        self.stop_condition = s
        self.run_condition = ru
        self.remove_condition = re
        TripleNode.create(
            self.agent.memory,
            subj=self.memid,
            pred_text="has_name",
            obj_text=self.__class__.__name__.lower(),
        )
        TaskNode(agent.memory, self.memid).update_task(task=self)
        # TODO if this is a command, put a chat_effect triple

    @staticmethod
    def step_wrapper(stepfn):
        def modified_step(self):
            if self.remove_condition.check():
                self.finished = True
            if self.finished:
                TaskNode(self.agent.memory, self.memid).get_update_status(
                    {"prio": -2, "finished": True}
                )
                return
            query = {
                "base_table": "Tasks",
                "base_range": {"minprio": 0.5},
                "triples": [{"pred_text": "_has_parent_task", "obj": self.memid}],
            }
            child_task_mems = self.agent.memory.basic_search(query)
            if child_task_mems:  # this task has active children, step them
                return
            r = stepfn(self)
            TaskNode(self.agent.memory, self.memid).update_task(task=self)
            return

        return modified_step

    def step(self):
        """The actual execution of a single step of the task is defined here."""
        pass

    def get_default_conditions(self, task_data, agent):
        """
        takes a task_data dict and fills in missing conditions with defaults

        Args:
            task_data (dict):  this function will try to use the values of "init_condition",
                               "stop_condition", "run_condition", and "remove_condition"
            agent (Droidlet Agent): the agent that is going to be doing the Task controlled by
                                    condition
            task (droidlet.shared_data_structs.Task):  the task to be controlled by the conditions
        """
        init_condition = task_data.get("init_condition", AlwaysCondition(None))

        run_condition = task_data.get("run_condition")
        stop_condition = task_data.get("stop_condition")
        if stop_condition is None:
            if run_condition is None:
                stop_condition = NeverCondition(None)
                run_condition = AlwaysCondition(None)
            else:
                stop_condition = NotCondition(run_condition)
        elif run_condition is None:
            run_condition = NotCondition(stop_condition)

        remove_condition = task_data.get("remove_condition", TaskStatusCondition(agent, self.memid))
        return init_condition, stop_condition, run_condition, remove_condition

    # FIXME remove all this its dead now...
    def interrupt(self):
        """Interrupt the task and set the flag"""
        self.interrupted = True

    def check_finished(self):
        """Check if the task has marked itself finished

        Returns:
            bool: If the task has finished
        """
        if self.finished:
            return self.finished

    def add_child_task(self, t, prio=1):
        TaskNode(self.agent.memory, self.memid).add_child_task(t, prio=prio)

    def __repr__(self):
        return str(type(self))


class Time:
    def __init__(self):
        self.init_time_raw = time.time()

    # converts from seconds to internal tick
    def round_time(self, t):
        return int(TICKS_PER_SEC * t)

    def get_time(self):
        return self.round_time(time.time() - self.init_time_raw)

    def get_world_hour(self):
        # returns a fraction of a day.  0 is sunrise, .5 is sunset, 1.0 is next day
        return (time.localtime()[3] - 8 + time.localtime()[4] / 60) / 24

    def add_tick(self, ticks=1):
        time.sleep(ticks / TICKS_PER_SEC)


class ErrorWithResponse(Exception):
    def __init__(self, chat):
        self.chat = chat


class NextDialogueStep(Exception):
    pass


class RGBDepth:
    """Class for the current RGB, depth and point cloud fetched from the robot.

    Args:
        rgb (np.array): RGB image fetched from the robot
        depth (np.array): depth map fetched from the robot
        pts (np.array [(x,y,z)]): array of x,y,z coordinates of the pointcloud corresponding
        to the rgb and depth maps.
    """

    rgb: np.array
    depth: np.array
    ptcloud: np.array

    def __init__(self, rgb, depth, pts):
        self.rgb = rgb
        self.depth = depth
        self.ptcloud = pts.reshape(rgb.shape)

    def get_pillow_image(self):
        return Image.fromarray(self.rgb, "RGB")

    def get_bounds_for_mask(self, mask):
        """for all points in the mask, returns the bounds as an axis-aligned bounding box.
        """
        if mask is None:
            return None
        points = self.ptcloud[np.where(mask == True)]
        points = xyz_pyrobot_to_canonical_coords(points)
        points = o3d.utility.Vector3dVector(points)
        obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        return np.concatenate([obb.get_min_bound(), obb.get_max_bound()])

    def get_coords_for_point(self, point):
        """fetches xyz from the point cloud in pyrobot coordinates and converts it to
        canonical world coordinates.
        """
        if point is None:
            return None
        xyz_p = self.ptcloud[point[1], point[0]]
        return xyz_pyrobot_to_canonical_coords(xyz_p)

    def to_struct(self, size=None, quality=10):
        import base64

        rgb = self.rgb
        depth = self.depth

        if size is not None:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_LINEAR)

        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = 255 - depth

        # webp seems to be better than png and jpg as a codec, in both compression and quality
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        fmt = ".webp"

        _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
        _, depth_data = cv2.imencode(fmt, depth, encode_param)
        return {
            "rgb": base64.b64encode(rgb_data).decode("utf-8"),
            "depth": base64.b64encode(depth_data).decode("utf-8"),
        }
