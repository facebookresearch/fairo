"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import random
import heuristic_perception
from typing import Tuple, Dict, Any, Optional, List
from word2number.w2n import word_to_num

import sys
import os

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.dialogue_objects import (
    TaskListWrapper,
    Interpreter,
    Say,
    SPEAKERLOOK,
    interpret_relative_direction,
    get_repeat_num,
    filter_by_sublocation,
)

from .schematic_helper import get_repeat_dir, interpret_schematic, interpret_size

from .facing_helper import FacingInterpreter

from .modify_helpers import (
    handle_fill,
    handle_rigidmotion,
    handle_scale,
    handle_replace,
    handle_thicken,
)
from .spatial_reasoning import ComputeLocations
from .block_helpers import get_block_type
from .condition_helper import MCConditionInterpreter
from .attribute_helper import MCAttributeInterpreter
from .point_target import PointTargetInterpreter
from base_agent.base_util import ErrorWithResponse
from base_agent.memory_nodes import PlayerNode
from mc_memory_nodes import MobNode, ItemStackNode
import dance
import tasks
from base_agent.task import ControlBlock
from mc_util import to_block_pos, Hole, XYZ


class MCInterpreter(Interpreter):
    """This class handles processes incoming chats and modifies the task stack

    Handlers should add/remove/reorder tasks on the stack, but not execute them.
    """

    def __init__(self, speaker: str, action_dict: Dict, **kwargs):
        super().__init__(speaker, action_dict, **kwargs)
        self.default_frame = "SPEAKER"
        self.workspace_memory_prio = ["Mob", "BlockObject"]
        self.subinterpret["attribute"] = MCAttributeInterpreter()
        self.subinterpret["condition"] = MCConditionInterpreter()
        self.subinterpret["specify_locations"] = ComputeLocations()
        self.subinterpret["facing"] = FacingInterpreter()
        self.subinterpret["point_target"] = PointTargetInterpreter()

        # logical_form --> possible task placed on stack + side effects
        self.action_handlers["BUILD"] = self.handle_build
        self.action_handlers["DESTROY"] = self.handle_destroy
        self.action_handlers["DIG"] = self.handle_dig
        self.action_handlers["SPAWN"] = self.handle_spawn
        self.action_handlers["FILL"] = self.handle_fill
        self.action_handlers["DANCE"] = self.handle_dance
        self.action_handlers["MODIFY"] = self.handle_modify
        ### FIXME generalize these
        self.action_handlers["GET"] = self.handle_get
        self.action_handlers["DROP"] = self.handle_drop

        self.task_objects = {
            "move": tasks.Move,
            "undo": tasks.Undo,
            "build": tasks.Build,
            "destroy": tasks.Destroy,
            "spawn": tasks.Spawn,
            "fill": tasks.Fill,
            "dig": tasks.Dig,
            "dance": tasks.Dance,
            "point": tasks.Point,
            "dancemove": tasks.DanceMove,
            "get": tasks.Get,
            "drop": tasks.Drop,
            "control": ControlBlock,
        }

    def handle_modify(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and handles a 'modify' command by either replying back or pushing 
        appropriate tasks to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        default_ref_d = {"filters": {"location": SPEAKERLOOK}}
        ref_d = d.get("reference_object", default_ref_d)
        # only modify blockobjects...
        objs = self.subinterpret["reference_objects"](
            self, speaker, ref_d, extra_tags=["_physical_object", "_voxel_object"]
        )
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to modify.")

        m_d = d.get("modify_dict")
        if not m_d:
            raise ErrorWithResponse(
                "I think you want me to modify an object but am not sure what to do"
            )
        tasks = []
        for obj in objs:
            if m_d["modify_type"] == "THINNER" or m_d["modify_type"] == "THICKER":
                destroy_task_data, build_task_data = handle_thicken(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "REPLACE":
                destroy_task_data, build_task_data = handle_replace(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "SCALE":
                destroy_task_data, build_task_data = handle_scale(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "RIGIDMOTION":
                destroy_task_data, build_task_data = handle_rigidmotion(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "FILL" or m_d["modify_type"] == "HOLLOW":
                destroy_task_data, build_task_data = handle_fill(self, speaker, m_d, obj)
            else:
                raise ErrorWithResponse(
                    "I think you want me to modify an object but am not sure what to do (parse error)"
                )

            if build_task_data:
                tasks.append(self.task_objects["build"](self.agent, build_task_data))

            if destroy_task_data:
                tasks.append(self.task_objects["build"](self.agent, destroy_task_data))

        return tasks, None, None

    def handle_spawn(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and handles a 'spawn' command by either replying back or 
        pushing a Spawn task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        # FIXME! use filters appropriately, don't search by hand
        spawn_triples = d.get("reference_object", {}).get("filters", {}).get("triples", [])
        if not spawn_triples:
            raise ErrorWithResponse("I don't understand what you want me to spawn.")
        names = [t.get("obj_text") for t in spawn_triples if t.get("pred_text", "") == "has_name"]
        if not any(names):
            raise ErrorWithResponse("I don't understand what you want me to spawn.")
        # if multiple possible has_name triples, just pick the first:
        object_name = names[0]
        schematic = self.memory.get_mob_schematic_by_name(object_name)
        if not schematic:
            raise ErrorWithResponse("I don't know how to spawn: %r." % (object_name))

        object_idm = list(schematic.blocks.values())[0]
        location_d = d.get("location", SPEAKERLOOK)
        mems = self.subinterpret["reference_locations"](self, speaker, location_d)
        steps, reldir = interpret_relative_direction(self, location_d)
        pos, _ = self.subinterpret["specify_locations"](self, speaker, mems, steps, reldir)
        repeat_times = get_repeat_num(d)
        tasks = []
        for i in range(repeat_times):
            task_data = {"object_idm": object_idm, "pos": pos, "action_dict": d}
            tasks.append(self.task_objects["spawn"](self.agent, task_data))
        return tasks, None, None

    def handle_build(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'build' command by either pushing a dialogue object
        or pushing a Build task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        # Get the segment to build
        if "reference_object" in d:
            # handle copy
            repeat = get_repeat_num(d)
            objs = self.subinterpret["reference_objects"](
                self,
                speaker,
                d["reference_object"],
                limit=repeat,
                extra_tags=["_voxel_object"],
                loose_speakerlook=True,
            )
            if len(objs) == 0:
                raise ErrorWithResponse("I don't understand what you want me to build")
            tagss = [
                [(p, v) for (_, p, v) in self.memory.get_triples(subj=obj.memid)] for obj in objs
            ]
            interprets = [
                [list(obj.blocks.items()), obj.memid, tags] for (obj, tags) in zip(objs, tagss)
            ]
        else:  # a schematic
            if d.get("repeat") is not None:
                repeat_dict = d
            else:
                repeat_dict = None
            interprets = interpret_schematic(
                self, speaker, d.get("schematic", {}), repeat_dict=repeat_dict
            )

        # Get the locations to build
        location_d = d.get("location", SPEAKERLOOK)
        mems = self.subinterpret["reference_locations"](self, speaker, location_d)
        steps, reldir = interpret_relative_direction(self, location_d)
        origin, offsets = self.subinterpret["specify_locations"](
            self,
            speaker,
            mems,
            steps,
            reldir,
            repeat_dir=get_repeat_dir(location_d),
            objects=interprets,
            enable_geoscorer=True,
        )
        interprets_with_offsets = [
            (blocks, mem, tags, off) for (blocks, mem, tags), off in zip(interprets, offsets)
        ]

        tasks_data = []
        for schematic, schematic_memid, tags, offset in interprets_with_offsets:
            og = np.array(origin) + offset
            task_data = {
                "blocks_list": schematic,
                "origin": og,
                "schematic_memid": schematic_memid,
                "schematic_tags": tags,
                "action_dict": d,
            }

            tasks_data.append(task_data)

        tasks = []
        for td in tasks_data:
            t = self.task_objects["build"](self.agent, td)
            tasks.append(t)
            print(self.agent.memory._db_read("SELECT uuid FROM Tasks WHERE uuid=?", t.memid))
        logging.info("Adding {} Build tasks to stack".format(len(tasks)))
        return tasks, None, None

    def handle_fill(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'fill' command by either pushing a dialogue object
        or pushing a Fill task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        tasks = []
        r = d.get("reference_object")
        if not r.get("filters"):
            r["filters"] = {"location", SPEAKERLOOK}

        # Get the reference location
        location_d = r["filters"].get("location", SPEAKERLOOK)
        mems = self.subinterpret["reference_locations"](self, speaker, location_d)
        steps, reldir = interpret_relative_direction(self, location_d)
        location, _ = self.subinterpret["specify_locations"](self, speaker, mems, steps, reldir)

        # Get nearby holes
        holes: List[Hole] = heuristic_perception.get_all_nearby_holes(self.agent, location)
        candidates: List[Tuple[XYZ, Hole]] = [
            (to_block_pos(np.mean(hole[0], axis=0)), hole) for hole in holes
        ]

        # Choose the best ones to fill
        repeat = get_repeat_num(d)
        holes = filter_by_sublocation(self, speaker, candidates, r, limit=repeat, loose=True)
        if holes is None:
            self.dialogue_stack.append_new(
                Say, "I don't understand what holes you want me to fill."
            )
            return tasks, None, None
        for hole in holes:
            _, hole_info = hole
            poss, hole_idm = hole_info
            # FIXME use filters properly...
            triples = d.get("triples", [])
            block_types = [
                t.get("obj_text") for t in triples if t.get("pred_text", "") == "has_block_type"
            ]
            try:
                fill_idm = get_block_type(block_types[0])
            except:
                fill_idm = hole_idm
            task_data = {"action_dict": d, "schematic": poss, "block_idm": fill_idm}

            tasks.append(self.task_objects["fill"](self.agent, task_data))

        if len(holes) > 1:
            self.dialogue_stack.append_new(Say, "Ok. I'll fill up the holes.")
        else:
            self.dialogue_stack.append_new(Say, "Ok. I'll fill that hole up.")

        return tasks, None, None

    def handle_destroy(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'destroy' command by either pushing a dialogue object
        or pushing a Destroy task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        default_ref_d = {"filters": {"location": SPEAKERLOOK}}
        ref_d = d.get("reference_object", default_ref_d)
        objs = self.subinterpret["reference_objects"](
            self, speaker, ref_d, extra_tags=["_destructible"]
        )
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to destroy.")

        # don't kill mobs
        if all(isinstance(obj, MobNode) for obj in objs):
            raise ErrorWithResponse("I don't kill animals, sorry!")
        if all(isinstance(obj, PlayerNode) for obj in objs):
            raise ErrorWithResponse("I don't kill players, sorry!")
        objs = [obj for obj in objs if not isinstance(obj, MobNode)]
        tasks = []
        for obj in objs:
            if hasattr(obj, "blocks"):
                schematic = list(obj.blocks.items())
                task_data = {"schematic": schematic, "action_dict": d}
                tasks.append(self.task_objects["destroy"](self.agent, task_data))
        logging.info("Added {} Destroy tasks to stack".format(len(tasks)))
        return tasks, None, None

    def handle_dig(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'dig' command by either pushing a dialogue object
        or pushing a Dig task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """

        def new_tasks():
            attrs = {}
            schematic_d = d.get("schematic", {"has_size": 2})
            # set the attributes of the hole to be dug.
            for dim, default in [("depth", 1), ("length", 1), ("width", 1)]:
                key = "has_{}".format(dim)
                if key in schematic_d:
                    attrs[dim] = word_to_num(schematic_d[key])
                elif "has_size" in schematic_d:
                    attrs[dim] = interpret_size(self, schematic_d["has_size"])
                else:
                    attrs[dim] = default
            # minecraft world is [z, x, y]
            padding = (attrs["depth"] + 4, attrs["length"] + 4, attrs["width"] + 4)
            location_d = d.get("location", SPEAKERLOOK)
            repeat_num = get_repeat_num(d)
            repeat_dir = get_repeat_dir(d)
            mems = self.subinterpret["reference_locations"](self, speaker, location_d)
            steps, reldir = interpret_relative_direction(self, location_d)
            origin, offsets = self.subinterpret["specify_locations"](
                self,
                speaker,
                mems,
                steps,
                reldir,
                repeat_num=repeat_num,
                repeat_dir=repeat_dir,
                padding=padding,
            )
            # add dig tasks in a loop
            tasks_todo = []
            for offset in offsets:
                og = np.array(origin) + offset
                t = self.task_objects["dig"](self.agent, {"origin": og, "action_dict": d, **attrs})
                tasks_todo.append(t)
            return tasks_todo

        if "stop_condition" in d:
            condition = self.subinterpret["condition"](self, speaker, d["stop_condition"])
            return (
                [
                    self.task_objects["control"](
                        self.agent,
                        data={
                            "new_tasks_fn": TaskListWrapper(new_tasks),
                            "stop_condition": condition,
                            "action_dict": d,
                        },
                    )
                ],
                None,
                None,
            )
        else:
            return new_tasks(), None, None

    def handle_dance(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'dance' command by either pushing a dialogue object
        or pushing a Dance task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """

        def new_tasks():
            repeat = get_repeat_num(d)
            tasks_to_do = []
            # only go around the x has "around"; FIXME allow other kinds of dances
            location_d = d.get("location")
            if location_d is not None:
                rd = location_d.get("relative_direction")
                if rd is not None and (
                    rd == "AROUND" or rd == "CLOCKWISE" or rd == "ANTICLOCKWISE"
                ):
                    ref_obj = None
                    location_reference_object = location_d.get("reference_object")
                    if location_reference_object:
                        objmems = self.subinterpret["reference_objects"](
                            self, speaker, location_reference_object
                        )
                        if len(objmems) == 0:
                            raise ErrorWithResponse("I don't understand where you want me to go.")
                        ref_obj = objmems[0]
                    for i in range(repeat):
                        refmove = dance.RefObjMovement(
                            self.agent,
                            ref_object=ref_obj,
                            relative_direction=location_d["relative_direction"],
                        )
                        t = self.task_objects["dance"](self.agent, {"movement": refmove})
                        tasks_to_do.append(t)
                    return tasks_to_do

            dance_type = d.get("dance_type", {"dance_type_name": "dance"})
            # FIXME holdover from old dict format
            if type(dance_type) is str:
                dance_type = dance_type = {"dance_type_name": "dance"}
            if dance_type.get("point"):
                target = self.subinterpret["point_target"](self, speaker, dance_type["point"])
                for i in range(repeat):
                    t = self.task_objects["point"](self.agent, {"target": target})
                    tasks_to_do.append(t)
            # MC bot does not control body turn separate from head
            elif dance_type.get("look_turn") or dance_type.get("body_turn"):
                lt = dance_type.get("look_turn") or dance_type.get("body_turn")
                f = self.subinterpret["facing"](self, speaker, lt)
                for i in range(repeat):
                    t = self.task_objects["dancemove"](self.agent, f)
                    tasks_to_do.append(t)
            else:
                if location_d is None:
                    dance_location = None
                else:
                    mems = self.subinterpret["reference_locations"](self, speaker, location_d)
                    steps, reldir = interpret_relative_direction(self, location_d)
                    dance_location, _ = self.subinterpret["specify_locations"](
                        self, speaker, mems, steps, reldir
                    )
                # TODO use name!
                if dance_type.get("dance_type_span") is not None:
                    dance_name = dance_type["dance_type_span"]
                    if dance_name == "dance":
                        dance_name = "ornamental_dance"
                    dance_memids = self.memory._db_read(
                        "SELECT DISTINCT(Dances.uuid) FROM Dances INNER JOIN Triples on Dances.uuid=Triples.subj WHERE Triples.obj_text=?",
                        dance_name,
                    )
                else:
                    dance_memids = self.memory._db_read(
                        "SELECT DISTINCT(Dances.uuid) FROM Dances INNER JOIN Triples on Dances.uuid=Triples.subj WHERE Triples.obj_text=?",
                        "ornamental_dance",
                    )
                dance_memid = random.choice(dance_memids)[0]
                dance_fn = self.memory.dances[dance_memid]
                for i in range(repeat):
                    dance_obj = dance.Movement(
                        agent=self.agent, move_fn=dance_fn, dance_location=dance_location
                    )
                    t = self.task_objects["dance"](self.agent, {"movement": dance_obj})
                    tasks_to_do.append(t)
            return tasks_to_do

        if "stop_condition" in d:
            condition = self.subinterpret["condition"](self, speaker, d["stop_condition"])
            return (
                [
                    self.task_objects["control"](
                        self.agent,
                        data={
                            "new_tasks_fn": TaskListWrapper(new_tasks),
                            "stop_condition": condition,
                            "action_dict": d,
                        },
                    )
                ],
                None,
                None,
            )
        else:
            return new_tasks(), None, None

    # FIXME this is not compositional/does not handle loops ("get all the x")
    def handle_get(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'get' command by either pushing a dialogue object
        or pushing a Get task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        ref_d = d.get("reference_object", None)
        if not ref_d:
            raise ErrorWithResponse("I don't understand what you want me to get.")

        objs = self.subinterpret["reference_objects"](
            self, speaker, ref_d, extra_tags=["_on_ground"]
        )
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to get.")
        obj = [obj for obj in objs if isinstance(obj, ItemStackNode)][0]
        item_stack = self.agent.get_item_stack(obj.eid)
        idm = (item_stack.item.id, item_stack.item.meta)
        task_data = {"idm": idm, "pos": obj.pos, "eid": obj.eid, "memid": obj.memid}
        return [self.task_objects["get"](self.agent, task_data)], None, None

    # FIXME this is not compositional/does not handle loops ("get all the x")
    def handle_drop(self, speaker, d) -> Tuple[Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'drop' command by either pushing a dialogue object
        or pushing a Drop task to the task stack. 

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        ref_d = d.get("reference_object", None)
        if not ref_d:
            raise ErrorWithResponse("I don't understand what you want me to drop.")

        objs = self.subinterpret["reference_objects"](
            self, speaker, ref_d, extra_tags=["_in_inventory"]
        )
        if len(objs) == 0:
            raise ErrorWithResponse("I don't understand what you want me to drop.")

        obj = [obj for obj in objs if isinstance(obj, ItemStackNode)][0]
        item_stack = self.agent.get_item_stack(obj.eid)
        idm = (item_stack.item.id, item_stack.item.meta)
        task_data = {"eid": obj.eid, "idm": idm, "memid": obj.memid}
        return [self.task_objects["drop"](self.agent, task_data)], None, None
        self.append_new_task(self.task_objects["drop"], task_data)
