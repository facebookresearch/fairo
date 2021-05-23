"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import logging
from typing import List
from droidlet.memory.memory_nodes import PlayerNode
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

    def update(self, agent):
        pass

    #################
    ###  Players  ###
    #################

    def update_other_players(self, player_list: List):
        # input is a list of player_structs from agent
        for p in player_list:
            mem = self.get_player_by_eid(p.entityId)
            if mem is None:
                memid = PlayerNode.create(self, p)
            else:
                memid = mem.memid
            PlayerNode.update(self, p, memid)

    ######################
    ###  DetectedObjs  ###
    ######################

    def get_detected_objects_tagged(self, *tags) -> List["DetectedObjectNode"]:
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        logging.info("get_detected_objects_tagged {}, tags {}".format(memids, tags))
        return memids

    ###############
    ###  Dances  ##
    ###############

    def add_dance(self, dance_fn, name=None, tags=[]):
        # a dance is movement determined as a sequence of steps, rather than by its destination
        return DanceNode.create(self, dance_fn, name=name, tags=tags)
