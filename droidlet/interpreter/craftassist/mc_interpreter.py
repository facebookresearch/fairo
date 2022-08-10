"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging

import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
from copy import deepcopy
from word2number.w2n import word_to_num

from droidlet.dialog.dialogue_task import Say

from droidlet.interpreter import (
    Interpreter,
    SPEAKERLOOK,
    interpret_relative_direction,
    filter_by_sublocation,
    interpret_dance_filter,
)

from .interpret_schematic import (
    get_repeat_dir,
    interpret_schematic,
    interpret_size,
    interpret_fill_schematic,
    interpret_mob_schematic,
)

from .interpret_facing import FacingInterpreter

from .interpret_modify import (
    handle_fill,
    handle_rigidmotion,
    handle_scale,
    handle_replace,
    handle_thicken,
)
from .spatial_reasoning import ComputeLocations
from ..interpret_conditions import ConditionInterpreter
from .interpret_attributes import MCAttributeInterpreter
from .point_target import PointTargetInterpreter
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.memory.memory_nodes import PlayerNode, TripleNode
from droidlet.memory.craftassist.mc_memory_nodes import MobNode, ItemStackNode
from droidlet.interpreter.craftassist import tasks, dance
from droidlet.task.task import ControlBlock, maybe_bundle_task_list


class MCInterpreter(Interpreter):
    """This class handles processes incoming chats and modifies the task stack

    Handlers should add/remove/reorder tasks on the stack, but not execute them.
    """

    def __init__(self, speaker, logical_form_memid, agent_memory, memid=None, low_level_data=None):
        super().__init__(speaker, logical_form_memid, agent_memory, memid=memid)
        self.default_frame = "SPEAKER"
        # These are coming from agent's low level
        self.block_data = low_level_data["block_data"]
        self.get_locs_from_entity = low_level_data["get_locs_from_entity"]
        self.special_shape_functions = low_level_data["special_shape_functions"]
        self.color_bid_map = low_level_data["color_bid_map"]
        self.allow_clarification = low_level_data["allow_clarification"]
        # These come from agent's perception
        self.get_all_holes_fn = low_level_data["get_all_holes_fn"]
        self.workspace_memory_prio = ["Mob", "BlockObject"]
        self.subinterpret["attribute"] = MCAttributeInterpreter()
        self.subinterpret["condition"] = ConditionInterpreter()
        self.subinterpret["specify_locations"] = ComputeLocations()
        self.subinterpret["facing"] = FacingInterpreter()
        self.subinterpret["dances_filters"] = interpret_dance_filter
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
        }

    def handle_modify(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
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
            self, speaker, ref_d, extra_tags=["_physical_object", "_VOXEL_OBJECT"]
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
                destroy_task_data, build_task_data = handle_replace(
                    self,
                    speaker,
                    m_d,
                    obj,
                    block_data=self.block_data,
                    color_bid_map=self.color_bid_map,
                )
            elif m_d["modify_type"] == "SCALE":
                destroy_task_data, build_task_data = handle_scale(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "RIGIDMOTION":
                destroy_task_data, build_task_data = handle_rigidmotion(self, speaker, m_d, obj)
            elif m_d["modify_type"] == "FILL" or m_d["modify_type"] == "HOLLOW":
                destroy_task_data, build_task_data = handle_fill(
                    self,
                    speaker,
                    m_d,
                    obj,
                    block_data=self.block_data,
                    color_bid_map=self.color_bid_map,
                )
            else:
                raise ErrorWithResponse(
                    "I think you want me to modify an object but am not sure what to do (parse error)"
                )

            if build_task_data:
                tasks.append(self.task_objects["build"](agent, build_task_data))

            if destroy_task_data:
                tasks.append(self.task_objects["build"](agent, destroy_task_data))

            return maybe_bundle_task_list(agent, tasks)

    def handle_spawn(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and handles a 'spawn' command by either replying back or
        pushing a Spawn task to the task stack.

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        # FIXME! use filters appropriately, don't search by hand
        filters_d = d.get("reference_object", {}).get("filters", {})
        object_idms = interpret_mob_schematic(self, speaker, filters_d)

        location_d = d.get("location", SPEAKERLOOK)
        locmems = self.subinterpret["reference_locations"](self, speaker, location_d)
        steps, reldir = interpret_relative_direction(self, location_d)
        pos, _ = self.subinterpret["specify_locations"](self, speaker, locmems, steps, reldir)
        tasks = []
        for i in range(len(object_idms)):
            task_data = {"object_idm": object_idms[i], "pos": pos, "action_dict": d}
            tasks.append(self.task_objects["spawn"](agent, task_data))
        return maybe_bundle_task_list(agent, tasks)

    def get_block_task_data(self, speaker, schematic_info, d):
        """
        takes schematic_info returned e.g. from interpret_schematic
        and the location logical form
        and returns task data for a list of build or dig tasks

        """
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
            objects=schematic_info,
        )
        schematic_info_with_offsets = [
            (blocks, mem, tags, off) for (blocks, mem, tags), off in zip(schematic_info, offsets)
        ]

        tasks_data = []
        for schematic, schematic_memid, tags, offset in schematic_info_with_offsets:
            og = np.array(origin) + offset
            task_data = {
                "blocks_list": schematic,
                "origin": og,
                "schematic_memid": schematic_memid,
                "schematic_tags": tags,
                "action_dict": d,
            }
            tasks_data.append(task_data)

        return tasks_data

    def handle_build(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
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
            ##########FIXME remove this when DSL updated!!!
            md = deepcopy(d)
            objs = self.subinterpret["reference_objects"](
                self,
                speaker,
                md["reference_object"],
                extra_tags=["_VOXEL_OBJECT"],
                loose_speakerlook=True,
            )
            if len(objs) == 0:
                raise ErrorWithResponse("I don't understand what you want me to build")
            tagss = [
                [
                    (p, v)
                    for (_, p, v) in self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                        self.memory, subj=obj.memid
                    )
                ]
                for obj in objs
            ]
            interprets = [
                [list(obj.blocks.items()), obj.memid, tags] for (obj, tags) in zip(objs, tagss)
            ]
        else:  # a schematic
            interprets = interpret_schematic(
                self,
                speaker,
                d.get("schematic", {}),
                self.block_data,
                self.color_bid_map,
                self.special_shape_functions,
            )

        tasks_data = self.get_block_task_data(speaker, interprets, d)
        tasks = []
        for td in tasks_data:
            t = self.task_objects["build"](agent, td)
            tasks.append(t)
        logging.info("Adding {} Build tasks to stack".format(len(tasks)))
        return maybe_bundle_task_list(agent, tasks)

    def handle_fill(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
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

        """
        FIXME: We need to fix this and perhaps put this in reasoning. Agent should run perception
        and put into memory. Interpreter shouldn't be perceiving, but should be able to
        ask the agent to do it when needed.
        """
        # Get nearby holes
        perception_holes = self.get_all_holes_fn(
            agent, location, self.block_data, agent.low_level_data["fill_idmeta"]
        )
        perception_output = CraftAssistPerceptionData(holes=perception_holes)
        output = self.memory.update(perception_output=perception_output)
        holes = output.get("holes", [])
        # Choose the best ones to fill
        holes = filter_by_sublocation(self, speaker, holes, r, loose=True)

        if holes is None:
            # FIXME: in stage III, replace agent with the lowlevel interface to sending chats
            raise ErrorWithResponse("I don't understand what holes you want me to fill.")
        tasks = []
        for hole in holes:
            poss = list(hole.blocks.keys())
            try:
                fill_memid = agent.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=hole.memid, pred_text="has_fill_type"
                )[0][2]
                fill_block_mem = self.memory.get_mem_by_id(fill_memid)
                fill_idm = (fill_block_mem.b, fill_block_mem.m)
            except:
                # FIXME use a constant name
                fill_idm = (3, 0)
            schematic, tags = interpret_fill_schematic(
                self,
                speaker,
                d.get("schematic", {}),
                poss,
                fill_idm,
                self.block_data,
                self.color_bid_map,
            )
            origin = np.min([xyz for (xyz, bid) in schematic], axis=0)
            task_data = {
                "blocks_list": schematic,
                "force": True,
                "origin": origin,
                "verbose": False,
                "embed": True,
                "fill_message": True,
                "schematic_tags": tags,
            }

            tasks.append(self.task_objects["build"](agent, task_data))

        if len(holes) > 1:
            Say(agent, task_data={"response_options": "Ok. I'll fill up the holes."})
        else:
            Say(agent, task_data={"response_options": "Ok. I'll fill that hole up."})

        return maybe_bundle_task_list(agent, tasks)

    def handle_destroy(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
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
                tasks.append(self.task_objects["destroy"](agent, task_data))
        logging.info("Added {} Destroy tasks to stack".format(len(tasks)))
        return maybe_bundle_task_list(agent, tasks)

    def handle_dig(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'dig' command by either pushing a dialogue object
        or pushing a Dig task to the task stack.

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """
        # FIXME assuming an "AND" here...
        # replace "has_name": "hole" to "has_name": "rectanguloid" for default
        schematic_where = (
            d.get("schematic", {}).get("filters", {}).get("where_clause", {}).get("AND", [])
        )
        for c in schematic_where:
            if c.get("pred_text") is not None and c.get("obj_text") == "hole":
                c["obj_text"] = "rectanguloid"

        interprets = interpret_schematic(
            self,
            speaker,
            d.get("schematic", {}),
            self.block_data,
            self.color_bid_map,
            self.special_shape_functions,
        )

        tasks_data = self.get_block_task_data(speaker, interprets, d)

        tasks = []
        for td in tasks_data:
            t = self.task_objects["dig"](agent, td)
            tasks.append(t)

        return maybe_bundle_task_list(agent, tasks)

    def handle_dance(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
        """This function reads the dictionary, resolves the missing details using memory
        and perception and handles a 'dance' command by either pushing a dialogue object
        or pushing a Dance task to the task stack.

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary
        """

        def new_tasks():
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
                    refmove = dance.RefObjMovement(
                        agent,
                        ref_object=ref_obj,
                        relative_direction=location_d["relative_direction"],
                    )
                    t = self.task_objects["dance"](agent, {"movement": refmove})
                    return t

            dance_type = d.get("dance_type", {})
            if dance_type.get("point"):
                target = self.subinterpret["point_target"](self, speaker, dance_type["point"])
                t = self.task_objects["point"](agent, {"target": target})
            # MC bot does not control body turn separate from head
            elif dance_type.get("look_turn") or dance_type.get("body_turn"):
                lt = dance_type.get("look_turn") or dance_type.get("body_turn")
                f = self.subinterpret["facing"](self, speaker, lt)
                t = self.task_objects["dancemove"](agent, f)
            else:
                if location_d is None:
                    dance_location = None
                else:
                    mems = self.subinterpret["reference_locations"](self, speaker, location_d)
                    steps, reldir = interpret_relative_direction(self, location_d)
                    dance_location, _ = self.subinterpret["specify_locations"](
                        self, speaker, mems, steps, reldir
                    )
                filters_d = dance_type.get("filters", {})
                filters_d["memory_type"] = "DANCES"
                F = self.subinterpret["filters"](self, speaker, dance_type.get("filters", {}))
                dance_memids, _ = F()
                # TODO correct selector in filters
                if dance_memids:
                    dance_memid = random.choice(dance_memids)
                    dance_mem = self.memory.get_mem_by_id(dance_memid)
                    dance_obj = dance.Movement(
                        agent=agent, move_fn=dance_mem.dance_fn, dance_location=dance_location
                    )
                    t = self.task_objects["dance"](agent, {"movement": dance_obj})
                else:
                    # dance out of scope
                    raise ErrorWithResponse("I don't know how to do that movement yet.")
            return t

        return new_tasks

    def handle_get(self, agent, speaker, d) -> Tuple[Any, Optional[str], Any]:
        """
        handles give, get bring, drop.
        arranges all pickups first, and then all dropoffs.

        Args:
            speaker: speaker_id or name.
            d: the complete action dictionary

        approach: find all reference objects given by d.get("reference_object")
        for each object, if it is not in agent's inventory,  set a task to pick it up
        then set a task to drop each of the reference objects.
        """
        # TODO: in grammar differentiate between
        # "get the ball you don't have and bring it to where you are now"
        # and
        # "drop the ball you have where you are now"
        # e.g. use an "in_my_inventory" fixed value

        ref_pickup = d.get("reference_object", None)
        d_receiver = d.get("receiver", None)

        tasks = []
        if ref_pickup is None and d_receiver is None:
            raise ErrorWithResponse(
                "I think you want me to get or put something, but I don't understand what"
            )

        if ref_pickup:
            objs = self.subinterpret["reference_objects"](self, speaker, ref_pickup)
            objs = [obj for obj in objs if isinstance(obj, ItemStackNode)]
            if not objs:
                raise ErrorWithResponse(
                    "I think you want me to get or put something, but I don't understand what"
                )
            # no selector, just pick the nearest object to agent from the list.  if agent is carrying one it should be nearest
            # FIXME: do this properly, add a nearest to agent selector
            if ref_pickup.get("filters", {}).get("selector") is None:
                min_dist_obj = None
                min_dist = 10000000
                for obj in objs:
                    dist = np.linalg.norm(np.array(obj.pos) - np.array(agent.pos))
                    if dist <= min_dist:
                        min_dist_obj = obj
                        min_dist = dist
                objs = [obj]

            for obj in objs:
                # some or all of the objects to deliver/drop may already be in inventory...
                # only pick up objects already on the ground.
                if "_on_ground" in obj.get_tags():
                    task_data = {
                        "pos": obj.pos,
                        "eid": obj.eid,
                        "obj_memid": obj.memid,
                    }
                    tasks.append(self.task_objects["get"](agent, task_data))

        if d_receiver:
            if len(objs) == 0:
                raise ErrorWithResponse("I don't understand what you want me to get or place.")
            try:
                if d_receiver.get("reference_object"):
                    # this is bring to an entity; but for now will just drop it at entities location
                    mems = self.subinterpret["reference_locations"](self, speaker, d_receiver)
                else:  # d_receiver.get("location"):
                    assert d_receiver.get("location") is not None
                    mems = self.subinterpret["reference_locations"](
                        self, speaker, d_receiver["location"]
                    )
            except NextDialogueStep:
                # TODO allow for clarification
                raise ErrorWithResponse("I don't understand where you want me to bring the thing.")
            # FIXME this should go in the ref_location subinterpret:
            steps, reldir = interpret_relative_direction(self, d_receiver)
            pos, _ = self.subinterpret["specify_locations"](self, speaker, mems, steps, reldir)
            # TODO: can this actually happen?
            if pos is None:
                raise ErrorWithResponse(
                    "I don't understand the location to which I am supposed to bring the thing."
                )

            for obj in objs:
                task_data = {"target": pos, "eid": obj.eid, "obj_memid": obj.memid}
                tasks.append(self.task_objects["drop"](agent, task_data))

            return maybe_bundle_task_list(agent, tasks)
