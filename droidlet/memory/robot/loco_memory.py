"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import logging
from typing import List
from collections import namedtuple
from droidlet.memory.memory_nodes import PlayerNode, TripleNode
from droidlet.memory.sql_memory import AgentMemory
from droidlet.memory.robot.loco_memory_nodes import *

SCHEMAS = [
    os.path.join(os.path.dirname(__file__), "..", "base_memory_schema.sql"),
    os.path.join(os.path.dirname(__file__), "loco_memory_schema.sql"),
]

# TODO "snapshot" memory type  (giving a what mob/object/player looked like at a fixed timestamp)
# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class LocoAgentMemory(AgentMemory):
    def __init__(
        self,
        db_file=":memory:",
        db_log_path=None,
        schema_paths=SCHEMAS,
        coordinate_transforms=None,
    ):
        super(LocoAgentMemory, self).__init__(
            db_file=db_file,
            schema_paths=schema_paths,
            db_log_path=db_log_path,
            nodelist=NODELIST,
            coordinate_transforms=coordinate_transforms,
        )
        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self._safe_pickle_saved_attrs = {}
        self.dances = {}

    ############################################
    ### Update world with perception updates ###
    ############################################

    def update(self, perception_output: namedtuple = None):
        """
        Updates the world with updates from agent's perception module.

        Args:
            perception_output: namedtuple with attributes -
                new_objects: List of new detections
                updated_objects: List of detections with updates
                humans: List of humans detected
        """
        if not perception_output:
            return

        # TODO there should be some sort of warning/error if self_pose is not updated
        if perception_output.self_pose is not None:
            x, z, yaw = perception_output.self_pose
            self.place_field.update_map(
                [
                    {
                        "pos": (x, 0, z),
                        "is_obstacle": True,
                        "memid": self.self_memid,
                        "is_move": True,
                    }
                ]
            )

        if perception_output.new_objects:
            for detection in perception_output.new_objects:
                memid = DetectedObjectNode.create(self, detection)
                # TODO use the bounds, not just the center
                pos = (
                    detection.get_xyz()["x"],
                    detection.get_xyz()["y"],
                    detection.get_xyz()["z"],
                )
                self.place_field.update_map([{"pos": pos, "memid": memid}])
        if perception_output.updated_objects:
            for detection in perception_output.updated_objects:
                memid = DetectedObjectNode.update(self, detection)
                # TODO use the bounds, not just the center
                pos = (
                    detection.get_xyz()["x"],
                    detection.get_xyz()["y"],
                    detection.get_xyz()["z"],
                )
                self.place_field.update_map([{"pos": pos, "memid": memid, "is_move": True}])
        if perception_output.humans:
            for human in perception_output.humans:
                HumanPoseNode.create(self, human)
                # FIXME, not putting in map, need to dedup?
        # FIXME make a proper diff.  what to do about discrepancies with objects?
        self.place_field.sync_traversible(perception_output.obstacle_map, h=0)

    #################
    ###  Players  ###
    #################

    def update_other_players(self, player_list: List):
        # input is a list of player_structs from agent
        for p in player_list:
            mem = self.nodes[PlayerNode.NODE_TYPE].get_player_by_eid(self, p.entityId)
            if mem is None:
                memid = PlayerNode.create(self, p)
            else:
                memid = mem.memid
            PlayerNode.update(self, p, memid)

    ######################
    ###  DetectedObjs  ###
    ######################

    def get_detected_objects_tagged(self, *tags) -> List["DetectedObjectNode"]:
        memids = set.intersection(
            *[set(self.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(self, t)) for t in tags]
        )
        logging.info("get_detected_objects_tagged {}, tags {}".format(memids, tags))
        return memids

    ###############
    ###  Dances  ##
    ###############

    def clear(self, objects):
        for o in objects:
            if o["memid"] != self.self_memid:
                self.forget(o["memid"])
