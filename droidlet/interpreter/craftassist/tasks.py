"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import time

from random import randint
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    PASSABLE_BLOCKS,
    BUILD_BLOCK_REPLACE_MAP,
    BUILD_IGNORE_BLOCKS,
    BUILD_INTERCHANGEABLE_PAIRS,
)
from droidlet.base_util import npy_to_blocks_list, blocks_list_to_npy, to_block_pos
from droidlet.shared_data_struct.craftassist_shared_utils import astar, MOBS_BY_ID
from droidlet.perception.craftassist.heuristic_perception import ground_height
from droidlet.lowlevel.minecraft.mc_util import manhat_dist, strip_idmeta

from droidlet.task.task import BaseMovementTask, Task
from droidlet.memory.memory_nodes import TaskNode, TripleNode
from droidlet.memory.craftassist.mc_memory_nodes import MobNode, ItemStackNode

# tasks should be interruptible; that is, if they
# store state, stopping the task and doing something
# else should not mess up their state and just the
# current state should be enough to do the task from
# any ob


class Dance(Task):
    """Perform a Dance task.

    Currently all dance movements are delegated to child dance move tasks.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> m = Movement(agent, move_fn)
        >>> task_data = {"movement": m}
        >>> dance = Dance(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        # movement should be a Movement object from dance.py
        self.movement = task_data.get("movement")
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        if self.finished:
            return
        self.interrupted = False
        mv = self.movement.get_move()
        if mv is None:
            self.finished = True
            return
        self.add_child_task(mv)


class DanceMove(Task):
    """Perform a Dance move task. Actual dance happens here.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
    """

    STEP_FNS = {
        (1, 0, 0): "step_pos_x",
        (-1, 0, 0): "step_neg_x",
        (0, 1, 0): "step_pos_y",
        (0, -1, 0): "step_neg_y",
        (0, 0, 1): "step_pos_z",
        (0, 0, -1): "step_neg_z",
    }

    def __init__(self, agent, task_data):
        super(DanceMove, self).__init__(agent)
        self.relative_yaw = task_data.get("relative_yaw")
        self.relative_pitch = task_data.get("relative_pitch")

        # look_turn is (yaw, pitch).  pitch = 0 is head flat
        self.head_yaw_pitch = task_data.get("head_yaw_pitch")
        self.head_xyz = task_data.get("head_xyz")

        self.translate = task_data.get("translate")
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        if self.finished:
            return
        # WARNING: these are in degrees to match MC and DSL;
        # memory stores angles in radians; but these are give by interpreter in angles
        # FIXME change DSL to use radians for fixed values and convert everything to radians
        if self.relative_yaw:
            agent.turn_angle(self.relative_yaw)
        if self.relative_pitch:
            agent.relative_head_pitch(self.relative_pitch)
        if self.head_xyz is not None:
            agent.look_at(self.head_xyz[0], self.head_xyz[1], self.head_xyz[2])
        elif self.head_yaw_pitch is not None:
            agent.set_look(self.head_yaw_pitch[0], self.head_yaw_pitch[1])

        if self.translate:
            step = self.STEP_FNS[self.translate]
            step_fn = getattr(agent, step)
            step_fn()
        else:
            # FIXME... do this right with ticks/timing
            time.sleep(0.1)

        self.finished = True


class Point(Task):
    """Perform a Point task.

    The pointed target is added to a queue for and agent will wait 200 ms to let pointing happen.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> target = [0, 0, 0, 2, 3, 4]
        >>> task_data = {"target": target}
        >>> point = Point(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super(Point, self).__init__(agent)
        self.target = task_data.get("target")
        self.start_time = agent.memory.get_time()
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        if self.finished:
            return
        if self.target is not None:
            self.agent.point_at(self.target)
            self.target = None
        if self.agent.memory.get_time() > self.start_time + 200:
            self.finished = True


class Move(BaseMovementTask):
    """Perform a Move task.

    Moving is stepped one block at a time. Agent can move towards one of the six directions:
        - positive of x axis
        - negative of x axis
        - positive of y axis
        - negative of y axis
        - positive of z axis
        - negative of z axis

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> target = (3, 7, 1)
        >>> task_data = {"target": target, "approx": 2}
        >>> move = Move(agent ,task_data)
    """

    STEP_FNS = {
        (1, 0, 0): "step_pos_x",
        (-1, 0, 0): "step_neg_x",
        (0, 1, 0): "step_pos_y",
        (0, -1, 0): "step_neg_y",
        (0, 0, 1): "step_pos_z",
        (0, 0, -1): "step_neg_z",
    }

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        if self.finished:
            return
        self.target = to_block_pos(np.array(task_data["target"]))
        self.approx = task_data.get("approx", 1)
        self.path = None
        self.replace = set()
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def target_to_memory(self, target):
        return to_block_pos(np.array(target))

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        if self.finished:
            return
        self.interrupted = False
        # replace blocks if possible
        R = self.replace.copy()
        self.replace.clear()
        for pos, idm in R:
            agent.set_held_item(idm)
            if agent.place_block(*pos):
                logging.debug("Move: replaced {}".format((pos, idm)))
            else:
                # try again later
                self.replace.add((pos, idm))
        if len(self.replace) > 0:
            logging.debug("Replace remaining: {}".format(self.replace))

        # check if finished
        if manhat_dist(tuple(agent.pos), self.target) <= self.approx:
            if len(self.replace) > 0:
                logging.error("Move finished with non-empty replace set: {}".format(self.replace))
            self.finished = True
            return

        # get path
        if self.path is None or tuple(agent.pos) != self.path[-1]:
            self.path = astar(agent, self.target, self.approx)
            if self.path is None:
                self.handle_no_path(agent)
                return

        # take a step on the path
        assert tuple(agent.pos) == self.path.pop()
        step = tuple(self.path[-1] - agent.pos)
        step_fn = getattr(agent, self.STEP_FNS[step])
        step_fn()

        self.last_stepped_time = agent.memory.get_time()

    def handle_no_path(self, agent):
        delta = self.target - agent.pos
        for vec, step_fn_name in self.STEP_FNS.items():
            if np.dot(delta, vec) > 0:
                newpos = agent.pos + vec
                x, y, z = newpos
                newpos_blocks = agent.get_blocks(x, x, y, y + 1, z, z)
                # dig if necessary
                for bp, idm in npy_to_blocks_list(newpos_blocks, newpos):
                    self.replace.add((bp, idm))
                    agent.dig(*bp)
                # move
                step_fn = getattr(agent, step_fn_name)
                step_fn()
                break

    def __repr__(self):
        return "<Move {} ±{}>".format(self.target, self.approx)


class Build(Task):
    """Perform a Build task.

    Agent will first clean up all blocks that needed to be removed, then start
    to build blocks. Both destroying and building operations are stepped one block
    at a time.

    The farthest block agent can destroy/build a three blocks away (by default).
    If a block is out of reach, a child move task will be added to task stack first.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.task_data = task_data
        self.embed = task_data.get("embed", False)
        self.schematic, _ = blocks_list_to_npy(task_data["blocks_list"])
        self.origin = task_data["origin"]
        self.verbose = task_data.get("verbose", True)
        self.force = task_data.get("force", False)
        self.attempts = 3 * np.ones(self.schematic.shape[:3], dtype=np.uint8)
        self.fill_message = task_data.get("fill_message", False)
        self.schematic_memid = task_data.get("schematic_memid", None)
        self.schematic_tags = task_data.get("schematic_tags", [])
        self.giving_up_message_sent = False
        self.wait = False
        self.old_blocks_list = None
        self.old_origin = None
        self.PLACE_REACH = task_data.get("PLACE_REACH", 3)

        # negative schematic related
        self.is_destroy_schm = task_data.get("is_destroy_schm", False)
        self.dig_message = task_data.get("dig_message", False)
        self.blockobj_memid = None
        self.DIG_REACH = task_data.get("DIG_REACH", 3)
        self.last_stepped_time = agent.memory.get_time()

        if self.is_destroy_schm:
            # is it destroying a whole block object? if so, save its tags
            self.destroyed_block_object_triples = []
            xyzs = set(strip_idmeta(task_data["blocks_list"]))
            mem = agent.memory.get_block_object_by_xyz(next(iter(xyzs)))
            # TODO what if there are several objects being destroyed?
            if mem and all(xyz in xyzs for xyz in mem.blocks.keys()):
                for pred in ["has_tag", "has_name", "has_colour"]:
                    self.destroyed_block_object_triples.extend(
                        agent.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                            agent.memory, subj=mem.memid, pred_text=pred
                        )
                    )
                logging.debug(
                    "Destroying block object {} tags={}".format(
                        mem.memid, self.destroyed_block_object_triples
                    )
                )

        # modify the schematic to avoid placing certain blocks
        for bad, good in BUILD_BLOCK_REPLACE_MAP.items():
            self.schematic[self.schematic[:, :, :, 0] == bad] = good
        self.new_blocks = []  # a list of (xyz, idm) of newly placed blocks

        # snap origin to ground if bottom level has dirt blocks
        # NOTE(kavyasrinet): except for when we are rebuilding the old dirt blocks, we
        # don't want to change the origin then, hence the self.force check.
        if not self.force and not self.embed and np.isin(self.schematic[:, :, :, 0], (2, 3)).any():
            h = ground_height(agent, self.origin, 0)
            self.origin[1] = h[0, 0]

        # get blocks occupying build area and save state for undo()
        ox, oy, oz = self.origin
        sy, sz, sx, _ = self.schematic.shape
        current = agent.get_blocks(ox, ox + sx - 1, oy, oy + sy - 1, oz, oz + sz - 1)
        self.old_blocks_list = npy_to_blocks_list(current, self.origin)
        if len(self.old_blocks_list) > 0:
            self.old_origin = np.min(strip_idmeta(self.old_blocks_list), axis=0)

        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        if self.finished:
            return
        self.interrupted = False
        # get blocks occupying build area
        ox, oy, oz = self.origin
        sy, sz, sx, _ = self.schematic.shape
        current = agent.get_blocks(ox, ox + sx - 1, oy, oy + sy - 1, oz, oz + sz - 1)

        # are we done?
        # TODO: diff ignores block meta right now because placing stairs and
        # chests in the appropriate orientation is non-trivial
        diff = (
            (current[:, :, :, 0] != self.schematic[:, :, :, 0])
            & (self.attempts > 0)
            & np.isin(current[:, :, :, 0], BUILD_IGNORE_BLOCKS, invert=True)
        )

        # ignore negative blocks if there is already air there
        diff &= (self.schematic[:, :, :, 0] + current[:, :, :, 0]) >= 0

        if self.embed:
            diff &= self.schematic[:, :, :, 0] != 0  # don't delete blocks if self.embed

        for pair in BUILD_INTERCHANGEABLE_PAIRS:
            diff &= np.isin(current[:, :, :, 0], pair, invert=True) | np.isin(
                self.schematic[:, :, :, 0], pair, invert=True
            )

        if not np.any(diff):
            self.finish(agent)
            return

        # blocks that would need to be removed
        remove_mask = diff & (current[:, :, :, 0] != 0)

        # destroy any blocks in the way (or any that are slated to be destroyed in schematic)
        # first
        rel_yzxs = np.argwhere(remove_mask)
        xyzs = set(
            [
                (x + self.origin[0], y + self.origin[1], z + self.origin[2])
                for (y, z, x) in rel_yzxs
            ]
        )
        if len(xyzs) != 0:
            self.remove_blocks(xyzs, agent)
            return
        # for a build task with destroy schematic,
        # it is done when all different blocks are removed.
        elif self.is_destroy_schm:
            self.finish(agent)
            return

        # get next block to place
        yzx = self.get_next_place_target(agent, current, diff)
        idm = self.schematic[tuple(yzx)]
        current_idm = current[tuple(yzx)]

        # try placing block
        target = yzx[[2, 0, 1]] + self.origin
        logging.debug("trying to place {} @ {}".format(idm, target))
        if tuple(target) in (tuple(agent.pos), tuple(agent.pos + [0, 1, 0])):
            # can't place block where you're standing, so step out of the way
            self.step_any_dir(agent)
            return
        if manhat_dist(agent.pos, target) <= self.PLACE_REACH:
            self.try_place_block(target, yzx, current_idm, idm, agent)
            return
        else:
            # too far to place; move first
            task = Move(agent, {"target": target, "approx": self.PLACE_REACH})
            self.add_child_task(task)

    def remove_blocks(self, xyzs, agent):
        logging.debug("Excavating {} blocks first".format(len(xyzs)))
        target = self.get_next_destroy_target(agent, xyzs)
        if target is None:
            logging.debug("No path from {} to {}".format(agent.pos, xyzs))
            agent.send_chat("There's no path, so I'm giving up")
            self.finished = True
            return

        if manhat_dist(agent.pos, target) <= self.DIG_REACH:
            success = agent.dig(*target)
            if success:
                self.perceive_removed_blocks(target, agent)
        else:
            mv = Move(agent, {"target": target, "approx": self.DIG_REACH})
            self.add_child_task(mv)

        return

    # FIXME, this should go in agent...
    # is being done here just so its easy to know agent placed block
    def perceive_removed_blocks(self, target, agent):
        agent.memory.maybe_remove_inst_seg(target)
        if self.is_destroy_schm:
            agent.memory.maybe_remove_block_from_memory(target, (0, 0), agent.areas_to_perceive)
        else:
            interesting, player_placed, agent_placed = agent.perception_modules[
                "low_level"
            ].mark_blocks_with_env_change(
                target,
                (0, 0),
                agent.low_level_data["boring_blocks"],
                agent_placed=True,
            )
            agent.memory.maybe_add_block_to_memory(
                interesting, player_placed, agent_placed, target, (0, 0)
            )
            self.add_tags(agent, (target, (0, 0)))
        # this is just clearing the changed blocks
        agent.get_changed_blocks()

    def try_place_block(self, target, yzx, current_idm, idm, agent):
        assert current_idm[0] != idm[0], "current={} idm={}".format(current_idm, idm)
        assert idm[0] > 0
        agent.set_held_item(idm)
        logging.debug("placing block {} @ {} from {}".format(idm, target, agent.pos))
        x, y, z = target.tolist()
        if agent.place_block(x, y, z):
            B = agent.get_blocks(x, x, y, y, z, z)
            if B[0, 0, 0, 0] == idm[0]:
                self.new_blocks.append((target, tuple(idm)))
                self.perceive_placed_block((x, y, z), idm, agent)
            else:
                logging.error(
                    "failed to place block {} @ {}, but place_block returned True. \
                        Got {} instead.".format(
                        idm, target, B[0, 0, 0, :]
                    )
                )
        else:
            logging.warn("failed to place block {} from {}".format(target, agent.pos))
        if idm[0] == 6:  # hacky: all saplings have id 6
            agent.set_held_item([351, 15])  # use bone meal on tree saplings
            if len(changed_blocks) > 0:
                sapling_pos = changed_blocks[0][0]
                x, y, z = sapling_pos
                for _ in range(6):  # use at most 6 bone meal (should be enough)
                    agent.use_item_on_block(x, y, z)
                    changed_blocks = agent.get_changed_blocks()
                    changed_block_poss = {block[0] for block in changed_blocks}
                    # sapling has grown to a full tree, stop using bone meal
                    if (x, y, z) in changed_block_poss:
                        break

        self.attempts[tuple(yzx)] -= 1
        if self.attempts[tuple(yzx)] == 0 and not self.giving_up_message_sent:
            agent.send_chat(
                "I'm skipping a block because I can't place it. Maybe something is in the way."
            )
            self.giving_up_message_sent = True

    # FIXME, this should go in agent...
    # is being done here just so its easy to know agent placed block
    def perceive_placed_block(self, target, idm, agent):
        interesting, player_placed, agent_placed = agent.perception_modules[
            "low_level"
        ].mark_blocks_with_env_change(
            target,
            tuple(idm),
            agent.low_level_data["boring_blocks"],
            agent_placed=True,
        )
        agent.memory.maybe_add_block_to_memory(
            interesting, player_placed, agent_placed, target, tuple(idm)
        )
        changed_blocks = agent.get_changed_blocks()
        self.add_tags(agent, (target, tuple(idm)))

    def add_tags(self, agent, block):
        # xyz, _ = npy_to_blocks_list(self.schematic, self.origin)[0]
        xyz = block[0]
        # this should not be an empty list- it is assumed the block passed in was just placed
        try:
            blockobj_memid = agent.memory.get_object_info_by_xyz(xyz, "BlockObjects")[0]
            self.blockobj_memid = blockobj_memid
        except:
            logging.debug(
                "Warning: place block returned true, but no block in memory after update"
            )
        TripleNode.create(
            agent.memory, subj=self.memid, pred_text="task_reference_object", obj=blockobj_memid
        )
        if self.schematic_memid:
            TripleNode.create(
                agent.memory,
                subj=blockobj_memid,
                pred_text="has_schematic",
                obj=self.schematic_memid,
            )
        if self.schematic_tags:
            for pred, obj in self.schematic_tags:
                TripleNode.create(agent.memory, subj=blockobj_memid, pred_text=pred, obj_text=obj)
                # sooooorrry  FIXME? when we handle triples better
                if "has_" in pred:
                    agent.memory.nodes[TripleNode.NODE_TYPE].tag(
                        agent.memory, self.blockobj_memid, obj
                    )

        agent.memory.nodes[TripleNode.NODE_TYPE].tag(agent.memory, blockobj_memid, "_in_progress")
        if self.dig_message:
            agent.memory.nodes[TripleNode.NODE_TYPE].tag(agent.memory, blockobj_memid, "hole")

    def finish(self, agent):
        if self.blockobj_memid is not None:
            agent.memory.nodes[TripleNode.NODE_TYPE].untag(
                agent.memory, self.blockobj_memid, "_in_progress"
            )
        if self.verbose:
            if self.is_destroy_schm:
                agent.send_chat("I finished destroying this")
            else:
                agent.send_chat("I finished building this")
        if self.fill_message:
            agent.send_chat("I finished filling this")
        if self.dig_message:
            agent.send_chat("I finished digging this.")
        self.finished = True

    def get_next_place_target(self, agent, current, diff):
        """Return the next block that will be targeted for placing

        In order:
        1. don't build over your own body
        2. build ground-up
        3. try failed blocks again at the end
        4. build closer blocks first

        Args:
        - current: yzxb-ordered current state of the region
        - diff: a yzx-ordered boolean mask of blocks that need addressing
        """
        relpos_yzx = (agent.pos - self.origin)[[1, 2, 0]]

        diff_yzx = list(np.argwhere(diff))
        diff_yzx.sort(key=lambda yzx: manhat_dist(yzx, relpos_yzx))  # 4
        diff_yzx.sort(key=lambda yzx: -self.attempts[tuple(yzx)])  # 3
        diff_yzx.sort(key=lambda yzx: yzx[0])  # 2
        diff_yzx.sort(
            key=lambda yzx: tuple(yzx) in (tuple(relpos_yzx), tuple(relpos_yzx + [1, 0, 0]))
        )  # 1
        return diff_yzx[0]

    def get_next_destroy_target(self, agent, xyzs):
        p = agent.pos
        for i, c in enumerate(sorted(xyzs, key=lambda c: manhat_dist(p, c))):
            path = astar(agent, c, approx=2)
            if path is not None:
                if i > 0:
                    logging.debug("Destroy get_next_destroy_target wasted {} astars".format(i))
                return c

        # No path to any of the blocks
        return None

    def step_any_dir(self, agent):
        px, py, pz = agent.pos
        B = agent.get_blocks(px - 1, px + 1, py - 1, py + 2, pz - 1, pz + 1)
        passable = np.isin(B[:, :, :, 0], PASSABLE_BLOCKS)
        walkable = passable[:-1, :, :] & passable[1:, :, :]  # head and feet passable
        assert walkable.shape == (3, 3, 3)
        relp = np.array([1, 1, 1])  # my pos is in the middle of the 3x3x3 cube
        for step, fn in (
            ((0, 1, 0), agent.step_pos_z),
            ((0, -1, 0), agent.step_neg_z),
            ((0, 0, 1), agent.step_pos_x),
            ((0, 0, -1), agent.step_neg_x),
            ((1, 0, 0), agent.step_pos_y),
            ((-1, 0, 0), agent.step_neg_y),
        ):
            if walkable[tuple(relp + step)]:
                fn()
                return
        raise Exception("Can't step in any dir from pos={} B={}".format(agent.pos, B))

    def undo(self, agent):
        schematic_tags = []
        if self.is_destroy_schm:
            # if rebuilding an object, get old object tags
            schematic_tags = [(pred, obj) for _, pred, obj in self.destroyed_block_object_triples]
            agent.send_chat("ok I will build it back.")
        else:
            agent.send_chat("ok I will remove it.")

        if self.old_blocks_list:
            self.add_child_task(
                Build(
                    agent,
                    {
                        "blocks_list": self.old_blocks_list,
                        "origin": self.old_origin,
                        "force": True,
                        "verbose": False,
                        "embed": self.embed,
                        "schematic_tags": schematic_tags,
                    },
                )
            )
        if len(self.new_blocks) > 0:
            self.add_child_task(Destroy(agent, {"schematic": self.new_blocks}))

    def __repr__(self):
        return "<Build {} @ {}>".format(len(self.schematic), self.origin)


class Fill(Task):
    """Perform a Fill task.

    It is done by delegated child build task.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> schematic = [(0, 0, 1), (0, 0, 2)]
        >>> task_data = {"schematic": schematic, "block_idm": (2, 0)}
        >>> fill = Fill(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.schematic = task_data["schematic"]  # a list of xyz tuples
        self.block_idm = task_data.get("block_idm", (2, 0))  # default 2: grass
        self.build_task = None
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()

        origin = np.min(self.schematic, axis=0)
        blocks_list = np.array([((x, y, z), self.block_idm) for (x, y, z) in self.schematic])

        build_task = Build(
            self.agent,
            {
                "blocks_list": blocks_list,
                "origin": origin,
                "force": True,
                "verbose": False,
                "embed": True,
                "fill_message": True,
            },
        )
        self.add_child_task(build_task)
        self.finished = True

    def undo(self, agent):
        query = "SELECT MEMORY FROM Task WHERE _has_parent_task=#={}".format(self.memid)
        _, build_mems = self.agent.memory.basic_search(query)
        if build_mems:
            build_mems[0].task.undo(agent)


class Destroy(Task):
    """Perform a Destroy task.

    The schematics to be detroyed are converted to negative counterpart first, then
    the actual destroy is delegated to a build task with the negative schematics.

    Args:
        agent (Agent): the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> schematic = [((2, 1, 0), (3, 0)), ((2, 2, 0), (3, 0))]
        >>> task_data = {"schematic": schematic, "dig_message": True}
        >>> destroy = Destroy(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.schematic = task_data["schematic"]  # list[(xyz, idm)]
        self.dig_message = True if "dig_message" in task_data else False
        self.submitted_build_task = False
        self.DIG_REACH = task_data.get("DIG_REACH", 3)
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        super().step()

        origin = np.min([(x, y, z) for ((x, y, z), (b, m)) in self.schematic], axis=0)

        def to_destroy_schm(block_list):
            """Convert idm of block list to negative

            For each block ((x, y, z), (b, m)), convert (b, m) to (-1, 0) indicating
            it should be digged or destroyed.

            Args:
            - block_list: a list of ((x,y,z), (id, meta))

            Returns:
            - a block list of ((x,y,z), (-1, 0))
            """

            destroy_schm = [((x, y, z), (-1, 0)) for ((x, y, z), (b, m)) in block_list]
            return destroy_schm

        destroy_schm = to_destroy_schm(self.schematic)
        if self.submitted_build_task:
            self.finished = True
        else:
            build_task = Build(
                self.agent,
                {
                    "blocks_list": destroy_schm,
                    "origin": origin,
                    "force": True,
                    "verbose": False,
                    "embed": True,
                    "dig_message": self.dig_message,
                    "is_destroy_schm": not self.dig_message,
                    "DIG_REACH": self.DIG_REACH,
                },
            )
            self.add_child_task(build_task)
            self.submitted_build_task = True

    def undo(self, agent):
        query = "SELECT MEMORY FROM Task WHERE _has_parent_task=#={}".format(self.memid)
        _, build_mems = self.agent.memory.basic_search(query)
        if build_mems:
            build_mems[0].task.undo(agent)


class Undo(Task):
    """Perform an Undo task.

    It is done through calling undo function of the specific task to be undone.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> task_data = {"memid": 2}
        >>> undo = Undo(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.to_undo_memid = task_data["memid"]
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        old_task_mem = self.agent.memory.get_mem_by_id(self.to_undo_memid)
        old_task_mem.task.undo(self.agent)
        self.finished = True

    def __repr__(self):
        return "<Undo {}>".format(self.to_undo_memid)


class Spawn(Task):
    """Perform a Spawn task.

    Note that it will try to spawn a mob in the given position and then check if there
    is any matching new mob within the certain area. It will not guarantee the mob will
    be spawned exactly once under the pessimistical circumstance.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> object_idm = (383, 90)
        >>> task_data = {"object_idm": object_idm, "pos": (2, 2, 2), "PLACE_REACH": 3}
        >>> spawn = Spawn(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.object_idm = task_data["object_idm"]
        self.mobtype = MOBS_BY_ID[self.object_idm[1]]
        self.pos = task_data["pos"]
        self.PLACE_REACH = task_data.get("PLACE_REACH", 3)
        self.last_stepped_time = agent.memory.get_time()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def find_nearby_new_mob(self, agent):
        mindist = 1000000
        near_new_mob = None
        x, y, z = self.pos
        y = y + 1
        for mob in agent.get_mobs():
            if MOBS_BY_ID[mob.mobType] == self.mobtype:
                dist = manhat_dist((mob.pos.x, mob.pos.y, mob.pos.z), (x, y, z))
                # hope this doesn;t take so long mob gets away...
                if dist < mindist:
                    #                    print(MOBS_BY_ID[mob.mobType], dist)
                    if not agent.memory.get_entity_by_eid(mob.entityId):
                        mindist = dist
                        near_new_mob = mob
        return mindist, near_new_mob

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        if manhat_dist(agent.pos, self.pos) > self.PLACE_REACH:
            task = Move(agent, {"target": self.pos, "approx": self.PLACE_REACH})
            self.add_child_task(task)
        else:
            agent.set_held_item(self.object_idm)
            if np.equal(self.pos, agent.pos).all():
                agent.step_neg_z()
            x, y, z = self.pos
            y = y + 1
            agent.place_block(x, y, z)
            time.sleep(0.1)
            mindist, placed_mob = self.find_nearby_new_mob(agent)
            if mindist < 3:
                memid = MobNode.create(agent.memory, placed_mob, agent_placed=True)
                mobmem = agent.memory.get_mem_by_id(memid)
                agent.memory.update_recent_entities(mems=[mobmem])
                if self.memid is not None:
                    agent.memory.nodes[TripleNode.NODE_TYPE].create(
                        agent.memory, subj=self.memid, pred_text="task_effect_", obj=mobmem.memid
                    )
                    # the chat_effect_ triple was already made when the task is added if there was a chat...
                    # but it points to the task memory.  link the chat to the mob memory:
                    chat_mem_triples = agent.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                        agent.memory, subj=None, pred_text="chat_effect_", obj=self.memid
                    )
                    if len(chat_mem_triples) > 0:
                        chat_memid = chat_mem_triples[0][0]
                        agent.memory.nodes[TripleNode.NODE_TYPE].create(
                            agent.memory,
                            subj=chat_memid,
                            pred_text="chat_effect_",
                            obj=mobmem.memid,
                        )
            self.finished = True


class Dig(Task):
    """Perform a Dig task.

    It is done by a delegated child destroy task with same schematics.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> task_data = {"ORIGIN": (0, 0, 1), "schematic": SCHEMATIC}
        >>> dig = Dig(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.origin = task_data["origin"]
        self.schematic = task_data["blocks_list"]

        self.destroy_task = None
        self.last_stepped_time = agent.memory.get_time()
        self.submitted_destroy = False

        TaskNode(agent.memory, self.memid).update_task(task=self)

    def undo(self, agent):
        if self.destroy_task is not None:
            self.destroy_task.undo(agent)

    @Task.step_wrapper
    def step(self):
        super().step()
        if not self.submitted_destroy:
            # HACK:  this needs to go in memory, not get called here:
            h = int(ground_height(self.agent, self.origin, 0)[0][0])

            schematic_ys = [l[1] for (l, idm) in self.schematic]
            H = max(schematic_ys)
            mx, my, mz = self.origin
            schematic = [
                ((l[0] + mx, h - H + l[1] - 1, l[2] + mz), (0, 0)) for (l, idm) in self.schematic
            ]
            destroy_task = Destroy(self.agent, {"schematic": schematic, "dig_message": True})
            self.submitted_destroy = True
            self.add_child_task(destroy_task)
        else:
            self.finished = True


class Get(Task):
    """Perform a Get task.

    Agent will first get the position of the item stacks to be gotten, then walk across the
    surrounding area to make sure the item stacks is gotten successfully.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> task_data = {"pos": (1, 0, 1), "eid": 11, "memid": 99}
        >>> dance = Dance(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.pos = task_data["pos"]
        self.eid = task_data["eid"]
        self.obj_memid = task_data["obj_memid"]
        self.approx = 1
        self.attempts = 20
        self.type_name = agent.memory.get_mem_by_id(self.obj_memid).type_name
        if getattr(agent, "backend", "cuberite") == "cuberite":
            self.idm = agent.low_level_data["block_data"]["name_to_bid"].get(self.type_name)
            if self.idm is None:
                raise Exception(
                    "I don't know how to pick that up: type_name={}".format(self.type_name)
                )
            self.item_count_before_get = agent.get_inventory_item_count(self.idm[0], self.idm[1])
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def add_move_to_item(self, agent):
        target = (self.pos[0] + randint(-1, 1), self.pos[1], self.pos[2] + randint(-1, 1))
        move_task = Move(agent, {"target": target, "approx": self.approx})
        self.add_child_task(move_task)

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        delta = 0
        close_to_item = np.linalg.norm(np.array(agent.pos) - np.array(self.pos)) < self.approx
        if getattr(agent, "backend", "cuberite") == "cuberite":
            delta = (
                agent.get_inventory_item_count(self.idm[0], self.idm[1])
                - self.item_count_before_get
            )
            if delta < 1:
                # in cuberite, just keep trying to walk around item, even if close
                self.add_move_to_item(agent)
        else:
            if not close_to_item:
                self.add_move_to_item(agent)
                return
            delta = agent.pick_entityId(self.eid)
        self.attempts -= 1
        if delta > 0:
            ItemStackNode.add_to_inventory(
                agent.memory, agent.memory.get_mem_by_id(self.obj_memid)
            )
            agent.send_chat("Got Item!")
            self.finished = True
            return

        if self.attempts == 0:
            agent.send_chat("I can't get this item. Giving up now")
            self.finished = True
            return

        return


class Drop(Task):
    """Perform a Drop task.

    Agent will drop the item stack on the ground then move away from where it stands right now
    to avoid getting the dropped item stacks automatically.

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data

    Examples::

        >>> task_data = {"eid": 11, "idm": (182, 0), "memid": 99}
        >>> dance = Dance(agent, task_data)
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        self.tries = 10
        self.target = task_data["target"]
        self.obj_memid = task_data["obj_memid"]
        self.eid = task_data["eid"]
        self.approx = 2
        self.type_name = agent.memory.get_mem_by_id(self.obj_memid).type_name
        if getattr(agent, "backend", "cuberite") == "cuberite":
            self.idm = agent.low_level_data["block_data"]["name_to_bid"].get(self.type_name)
            if self.idm is None:
                raise Exception(
                    "I don't know how to drop that: type_name={}".format(self.type_name)
                )
        TaskNode(agent.memory, self.memid).update_task(task=self)

    def get_memids_by_eid(self, memory, eid):
        cmd = "SELECT MEMORY FROM ReferenceObject WHERE eid={}".format(eid)
        memids, _ = memory.basic_search(cmd)
        return memids

    def find_nearby_new_item_stack(self, agent, id, meta):
        mindist = 3
        near_new_item_stack = None
        for item_stack in agent.get_item_stacks():
            if item_stack.item.id == id and item_stack.item.meta == meta:
                dist = np.linalg.norm(np.array(agent.get_player().pos) - np.array(item_stack.pos))
                if dist < mindist:
                    mindist = dist
                    near_new_item_stack = item_stack

        return mindist, near_new_item_stack

    @Task.step_wrapper
    def step(self):
        super().step()
        agent = self.agent
        if self.finished:
            return
        if not self.get_memids_by_eid(agent.memory, self.eid):
            agent.send_chat("I can't find it in my inventory!")
            self.finished = False
            return
        dist_to_target = np.linalg.norm(np.array(agent.pos) - np.array(self.target))
        close_to_target = dist_to_target <= self.approx
        if not close_to_target:
            move_task = Move(agent, {"target": self.target, "approx": self.approx})
            self.add_child_task(move_task)
            self.tries = self.tries - 1
            return

        node = agent.memory.get_mem_by_id(self.obj_memid)
        count = node.count
        if getattr(agent, "backend", "cuberite") == "cuberite":
            id, m = self.idm
            dropped_item_stack = agent.drop_inventory_item_stack(id, m, count)
            time.sleep(0.05)
            mindist, dropped_item_stack = self.find_nearby_new_item_stack(agent, id, m)
        else:
            dropped_item_stack = agent.drop(self.eid)
        if dropped_item_stack:
            ItemStackNode.remove_from_inventory(agent.memory, node)
            if getattr(agent, "backend", "cuberite") == "cuberite":
                new_eid = dropped_item_stack.entityId
                if new_eid != self.eid:
                    ItemStackNode.update_item_stack_eid(agent.memory, self.obj_memid, new_eid)
                ItemStackNode.maybe_update_item_stack_position(agent.memory, dropped_item_stack)
            agent.memory.nodes[TripleNode.NODE_TYPE].tag(
                agent.memory, self.obj_memid, "_on_ground"
            )
            agent.send_chat("here it is".format(node.type_name))
        else:
            agent.send_chat("I tried to drop the {}, but I can't".format(node.type_name))

        self.finished = True
