"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys

import numpy as np
import logging
from collections import Counter
from typing import cast, List, Sequence, Dict

from .mc_util import XYZ, LOOK, POINT_AT_TARGET, IDM, Block
from . import minecraft_specs
from .entities import MOBS_BY_ID


from base_agent.memory_nodes import link_archive_to_mem, ReferenceObjectNode, MemoryNode, NODELIST


class VoxelObjectNode(ReferenceObjectNode):
    """This is a reference object that is distributed over
    multiple voxels and uses VoxelObjects table to hold the
    location of the voxels; and ReferenceObjects to hold 'global' info
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        locs (tuple): List of (x, y, z) tuples
        blocks (dict): Dictionary of (x, y, z) -> (blockid, meta)
        update_times (dict): Dictionary of (x, y, z) -> time this was last updated
        player_placed (dict): Dictionary of (x, y, z) -> was this placed by player ?
        agent_placed (dict): Dictionary of (x, y, z) -> was this placed by the agent ?
    
    Examples::
        >>> node_list = [TaskNode, VoxelObjectNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> VoxelObjectNode(agent_memory=agent_memory, memid=memid)
    """

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        ref = self.agent_memory._db_read("SELECT * FROM ReferenceObjects WHERE uuid=?", self.memid)
        if len(ref) == 0:
            raise Exception("no mention of this VoxelObject in ReferenceObjects Table")
        self.ref_info = ref[0]
        voxels = self.agent_memory._db_read("SELECT * FROM VoxelObjects WHERE uuid=?", self.memid)
        self.locs: List[tuple] = []
        self.blocks: Dict[tuple, tuple] = {}
        self.update_times: Dict[tuple, int] = {}
        self.player_placed: Dict[tuple, bool] = {}
        self.agent_placed: Dict[tuple, bool] = {}
        for v in voxels:
            loc = (v[1], v[2], v[3])
            self.locs.append(loc)
            if v[4]:
                assert v[5] is not None
                self.blocks[loc] = (v[4], v[5])
            else:
                self.blocks[loc] = (None, None)
            self.agent_placed[loc] = v[6]
            self.player_placed[loc] = v[7]
            self.update_times[loc] = v[8]
            # TODO assert these all the same?
            self.memtype = v[9]

    def get_pos(self) -> XYZ:
        return cast(XYZ, tuple(int(x) for x in np.mean(self.locs, axis=0)))

    def get_point_at_target(self) -> POINT_AT_TARGET:
        point_min = [int(x) for x in np.min(self.locs, axis=0)]
        point_max = [int(x) for x in np.max(self.locs, axis=0)]
        return cast(POINT_AT_TARGET, point_min + point_max)

    def get_bounds(self):
        M = np.max(self.locs, axis=0)
        m = np.min(self.locs, axis=0)
        return m[0], M[0], m[1], M[1], m[2], M[2]

    def snapshot(self, agent_memory):
        archive_memid = self.new(agent_memory, snapshot=True)
        for loc in self.locs:
            cmd = "INSERT INTO ArchivedVoxelObjects (uuid, x, y, z, bid, meta, agent_placed, player_placed, updated, ref_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            values = (
                archive_memid,
                loc[0],
                loc[1],
                loc[2],
                self.blocks[loc][0],
                self.blocks[loc][1],
                self.agent_placed[loc],
                self.player_placed[loc],
                self.update_times[loc],
                self.memtype,
            )
            agent_memory.db_write(cmd, *values)

        archive_memid = self.new(agent_memory, snapshot=True)
        cmd = "INSERT INTO ArchivedReferenceObjects (uuid, eid, x, y, z, yaw, pitch, name, type_name, ref_type, player_placed, agent_placed, created, updated, voxel_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        info = list(self.ref_info)
        info[0] = archive_memid
        agent_memory.db_write(cmd, *info)
        link_archive_to_mem(agent_memory, self.memid, archive_memid)
        return archive_memid


class BlockObjectNode(VoxelObjectNode):
    """This is a voxel object that represents a set of physically present blocks.
    it is considered to be nonephemeral
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Examples::
        >>> node_list = [TaskNode, BlockObjectNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> BlockObjectNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = [
        "uuid",
        "x",
        "y",
        "z",
        "bid",
        "meta",
        "agent_placed",
        "player_placed",
        "updated",
    ]
    TABLE = "ReferenceObjects"
    NODE_TYPE = "BlockObject"

    @classmethod
    def create(cls, memory, blocks: Sequence[Block]) -> str:
        """Creates a new entry into the ReferenceObjects table
        Returns:
            string: memid of the entry
        Examples::
            >>> memory = AgentMemory()
            >>> blocks = [((1, 0, 34), (10, 1)), ((1, 0, 35), (10, 1)),
                          ((2, 0, 34), (2, 2)), ((3, 0, 34), (10, 0))]
            >>> create(memory, blocks)
        """
        # check if block object already exists in memory
        for xyz, _ in blocks:
            old_memids = memory.get_block_object_ids_by_xyz(xyz)
            if old_memids:
                return old_memids[0]
        memid = cls.new(memory)
        # TODO check/assert this isn't there...
        cmd = "INSERT INTO ReferenceObjects (uuid, x, y, z, ref_type, voxel_count) VALUES ( ?, ?, ?, ?, ?, ?)"
        # TODO this is going to cause a bug, need better way to initialize and track mean loc
        memory.db_write(cmd, memid, 0, 0, 0, "BlockObjects", 0)
        for block in blocks:
            memory.upsert_block(block, memid, "BlockObjects")
        memory.tag(memid, "_block_object")
        memory.tag(memid, "VOXEL_OBJECT")
        memory.tag(memid, "_physical_object")
        memory.tag(memid, "_destructible")
        # this is a hack until memory_filters does "not"
        memory.tag(memid, "_not_location")
        logging.debug(
            "Added block object {} with {} blocks, {}".format(
                memid, len(blocks), Counter([idm for _, idm in blocks])
            )
        )

        return memid

    def __repr__(self):
        return "<BlockObject Node @ {}>".format(list(self.blocks.keys())[0])


# note: instance segmentation objects should not be tagged except by the creator
# build an archive if you want to tag permanently
class InstSegNode(VoxelObjectNode):
    """This is a voxel object that represents a region of space, 
    and is considered ephemeral
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        locs (tuple): List of (x, y, z) tuples for this object
        blocks (dict): Dictionary of (x, y, z) to (blockid, meta) 
        tags (list): List of tags for this object
    
    Examples::
        >>> node_list = [TaskNode, InstSegNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> InstSegNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "x", "y", "z", "ref_type"]
    TABLE = "ReferenceObjects"
    NODE_TYPE = "InstSeg"

    @classmethod
    def create(cls, memory, locs, tags=[]) -> str:
        """Creates a new entry into the VoxelObjects table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> locs = [(1, 0, 34), (1, 0, 35), (2, 0, 34), (3, 0, 34)]
            >>> tags = ["shiny", "bright"]
            >>> create(memory, locs=locs, tags=tags)
        """
        # TODO option to not overwrite
        # check if instance segmentation object already exists in memory
        inst_memids = {}
        for xyz in locs:
            m = memory._db_read(
                'SELECT uuid from VoxelObjects WHERE ref_type="inst_seg" AND x=? AND y=? AND z=?',
                *xyz
            )
            if len(m) > 0:
                for memid in m:
                    inst_memids[memid[0]] = True
        # FIXME just remember the locs in the first pass
        for m in inst_memids.keys():
            olocs = memory._db_read("SELECT x, y, z from VoxelObjects WHERE uuid=?", m)
            # TODO maybe make an archive?
            if len(set(olocs) - set(locs)) == 0:
                memory.forget(m)

        memid = cls.new(memory)
        loc = np.mean(locs, axis=0)
        # TODO check/assert this isn't there...
        cmd = "INSERT INTO ReferenceObjects (uuid, x, y, z, ref_type) VALUES ( ?, ?, ?, ?, ?)"
        memory.db_write(cmd, memid, loc[0], loc[1], loc[2], "inst_seg")
        for loc in locs:
            cmd = "INSERT INTO VoxelObjects (uuid, x, y, z, ref_type) VALUES ( ?, ?, ?, ?, ?)"
            memory.db_write(cmd, memid, loc[0], loc[1], loc[2], "inst_seg")
        memory.tag(memid, "VOXEL_OBJECT")
        memory.tag(memid, "_inst_seg")
        memory.tag(memid, "_destructible")
        # this is a hack until memory_filters does "not"
        memory.tag(memid, "_not_location")
        for tag in tags:
            if type(tag) is str:
                memory.tag(memid, tag)
            elif type(tag) is dict:
                for k, v in tag.items():
                    memory.add_triple(subj=memid, pred_text=k, obj_text=v)
        return memid

    def __init__(self, memory, memid: str):
        super().__init__(memory, memid)
        r = memory._db_read("SELECT x, y, z FROM VoxelObjects WHERE uuid=?", self.memid)
        self.locs = r
        self.blocks = {l: (0, 0) for l in self.locs}
        tags = memory.get_triples(subj=self.memid, pred_text="has_tag")
        self.tags = []  # noqa: T484
        for tag in tags:
            if tag[2][0] != "_":
                self.tags.append(tag[2])

    def __repr__(self):
        return "<InstSeg Node @ {} with tags {} >".format(self.locs, self.tags)


class MobNode(ReferenceObjectNode):
    """This is a memory node representing a mob (moving object) in game
    
    Args:
        agent_memory  (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        eid (int): Entity ID of the mob node
        pos (tuple): (x, y, z) coordinates of the mob
        look (tuple): (yaw, pitch) of the mob
    
    Examples::
        >>> node_list = [TaskNode, MobNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> MobNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = [
        "uuid",
        "eid",
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "ref_type",
        "type_name",
        "player_placed",
        "agent_placed",
        "created",
    ]
    TABLE = "ReferenceObjects"
    NODE_TYPE = "Mob"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, x, y, z, yaw, pitch = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z, yaw, pitch FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.pos = (x, y, z)
        self.look = (yaw, pitch)

    @classmethod
    def create(cls, memory, mob, player_placed=False, agent_placed=False) -> str:
        """Creates a new entry into the ReferenceObjects table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> from entities import MOBS_BY_ID
            >>> chicken = {v: k for k, v in MOBS_BY_ID.items()}["chicken"]
            >>> mob_id, mob_type, pos, look = 42, chicken, Pos(3, 4, 5), Look(0.0, 0.0)
            >>> mob = Mob(mob_id, mob_type, pos, look)) # get an instance of the Mob class
            >>> player_placed=True # spawned by player
            >>> create(memory, mob, player_placed=player_placed)
        """
        # TODO warn/error if mob already in memory?
        memid = cls.new(memory)
        mobtype = MOBS_BY_ID[mob.mobType]
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, x, y, z, yaw, pitch, ref_type, type_name, player_placed, agent_placed, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            memid,
            mob.entityId,
            mob.pos.x,
            mob.pos.y,
            mob.pos.z,
            mob.look.yaw,
            mob.look.pitch,
            "mob",
            mobtype,
            player_placed,
            agent_placed,
            memory.get_time(),
        )
        memory.tag(memid, "mob")
        memory.tag(memid, "_physical_object")
        memory.tag(memid, "_animate")
        # this is a hack until memory_filters does "not"
        memory.tag(memid, "_not_location")
        memory.tag(memid, mobtype)
        memory.add_triple(subj=memid, pred_text="has_name", obj_text=mobtype)
        return memid

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    def get_look(self) -> LOOK:
        yaw, pitch = self.agent_memory._db_read_one(
            "SELECT yaw, pitch FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.look = (yaw, pitch)
        return self.look

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        # use the block above the mob as point_at_target
        return cast(POINT_AT_TARGET, (x, y + 1, z, x, y + 1, z))

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z


class ItemStackNode(ReferenceObjectNode):
    """A memory node for an item stack, which is something on the ground, 
    this is different from the placed blocks and can be picked up by the player/agent 
    if they are close to it.
    
    Args:
        agent_memory  (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        memid (string): MemoryID of the node
        eid (int): Entity ID of the item
        pos(tuple): (x, y, z) coordinates of the item
    
    Examples::
        >>> node_list = [TaskNode, ItemStackNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> ItemStackNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_ROWS = ["uuid", "eid", "x", "y", "z", "type_name", "ref_type", "created"]
    TABLE = "ReferenceObjects"
    NODE_TYPE = "ItemStack"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, x, y, z = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.memid = memid
        self.eid = eid
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory, item_stack) -> str:
        """Creates a new entry into the ReferenceObjects table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> from collections import namedtuple
            >>> ItemStack = namedtuple("ItemStack", "entityId, pos")
            >>> item_stack = ItemStack(12345678, Pos(0.0, 0.0, 0.0))
            >>> create(memory, item_stack)
        """
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]
        type_name = bid_to_name[(item_stack.item.id, item_stack.item.meta)]
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, x, y, z, type_name, ref_type, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            memid,
            item_stack.entityId,
            item_stack.pos.x,
            item_stack.pos.y,
            item_stack.pos.z,
            type_name,
            "item_stack",
            memory.get_time(),
        )
        memory.tag(memid, type_name)
        memory.tag(memid, "_item_stack")
        memory.tag(memid, "_on_ground")
        memory.tag(memid, "_physical_object")
        # this is a hack until memory_filters does "not"
        memory.tag(memid, "_not_location")
        return memid

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        # use the block above the item stack as point_at_target
        return cast(POINT_AT_TARGET, (x, y + 1, z, x, y + 1, z))

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z


class SchematicNode(MemoryNode):
    """A memory node representing a plan for an object that could 
    be built in the environment.
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        blocks (dict): Mapping of each (x, y, z) coordinate to the (block_id, meta) of
                the block at that coordinate.
    
    Examples::
        >>> node_list = [TaskNode, SchematicNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> SchematicNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "x", "y", "z", "bid", "meta"]
    TABLE = "Schematics"
    NODE_TYPE = "Schematic"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        if memid in agent_memory.schematics.keys():
            self.blocks = {
                (x, y, z): (b, m) for ((x, y, z), (b, m)) in agent_memory.schematics[memid]
            }
        else:
            r = self.agent_memory._db_read(
                "SELECT x, y, z, bid, meta FROM Schematics WHERE uuid=?", self.memid
            )
            self.blocks = {(x, y, z): (b, m) for (x, y, z, b, m) in r}

    @classmethod
    def create(cls, memory, blocks: Sequence[Block]) -> str:
        """Creates a new entry into the Schematics table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> blocks = [((0, 0, 1), (1, 0)), ((0, 0, 2), (1, 0)),
                          ((0, 0, 3), (2, 0))]
            >>> create(memory, blocks)
        """
        memid = cls.new(memory)
        for ((x, y, z), (b, m)) in blocks:
            memory.db_write(
                """
                    INSERT INTO Schematics(uuid, x, y, z, bid, meta)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                memid,
                x,
                y,
                z,
                b,
                m,
            )
        return memid


class BlockTypeNode(MemoryNode):
    """This is a memory node representing the type of a block in Minecraft
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        type_name (string): Name of the type of block (example: wool)
        b (int): The id of the block
        m (int): The meta information of a block
    
    Examples::
        >>> node_list = [TaskNode, BlockTypeNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> BlockTypeNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "type_name", "bid", "meta"]
    TABLE = "BlockTypes"
    NODE_TYPE = "BlockType"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        type_name, b, m = self.agent_memory._db_read(
            "SELECT type_name, bid, meta FROM BlockTypes WHERE uuid=?", self.memid
        )[0]
        self.type_name = type_name
        self.b = b
        self.m = m

    @classmethod
    def create(cls, memory, type_name: str, idm: IDM) -> str:
        """Creates a new entry into the BlockTypes table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> type_name = "air_block"
            >>> idm = (0, 0)
            >>> create(memory, type_name, idm)
        """
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO BlockTypes(uuid, type_name, bid, meta) VALUES (?, ?, ?, ?)",
            memid,
            type_name,
            idm[0],
            idm[1],
        )
        return memid


class MobTypeNode(MemoryNode):
    """This represents a mob type memory node (the type of a mob, 
    example: animal)
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        type_name (string): Name of the mob type
        b (int): Id of the mob type
        m (int): Meta information of the mob type
    
    Examples::
        >>> node_list = [TaskNode, MobTypeNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> MobTypeNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "type_name", "bid", "meta"]
    TABLE = "MobTypes"
    NODE_TYPE = "MobType"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        type_name, b, m = self.agent_memory._db_read(
            "SELECT type_name, bid, meta FROM MobTypes WHERE uuid=?", self.memid
        )
        self.type_name = type_name
        self.b = b
        self.m = m

    @classmethod
    def create(cls, memory, type_name: str, idm: IDM) -> str:
        """Creates a new entry into the MobTypes table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> type_name = "spawn husk"
            >>> idm = (23, 0)
            >>> create(memory, type_name, idm)
        """
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO MobTypes(uuid, type_name, bid, meta) VALUES (?, ?, ?, ?)",
            memid,
            type_name,
            idm[0],
            idm[1],
        )
        return memid


class DanceNode(MemoryNode):
    """This is a memory node representing a dance or sequence of movement steps
    
    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node
    
    Attributes:
        dance_fn (function): The function representing the execution of the dance
    
    Examples::
        >>> node_list = [TaskNode, DanceNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> DanceNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid"]
    TABLE = "Dances"
    NODE_TYPE = "Dance"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO put in DB/pickle like tasks?
        self.dance_fn = self.agent_memory.dances[memid]

    @classmethod
    def create(cls, memory, dance_fn, name=None, tags=[]) -> str:
        """Creates a new entry into the Dances table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> from dance import *
            >>> konami_dance = [
                    {"translate": (0, 1, 0)},
                    {"translate": (0, 1, 0)},
                    {"translate": (0, -1, 0)},
                    {"translate": (0, -1, 0)},
                    {"translate": (0, 0, -1)},
                    {"translate": (0, 0, 1)},
                    {"translate": (0, 0, -1)},
                    {"translate": (0, 0, 1)},
                ]
            >>> dance_fn = generate_sequential_move_fn(konami_dance)
            >>> name = "konami_dance"
            >>> tags = ["dance", "konami"]
            >>> create(memory, dance_fn, name=name, tags=tags)
        """
        memid = cls.new(memory)
        memory.db_write("INSERT INTO Dances(uuid) VALUES (?)", memid)
        # TODO put in db via pickle like tasks?
        memory.dances[memid] = dance_fn
        if name is not None:
            memory.add_triple(subj=memid, pred_text="has_name", obj_text=name)
        if len(tags) > 0:
            for tag in tags:
                memory.add_triple(subj=memid, pred_text="has_tag", obj_text=tag)
        return memid


class RewardNode(MemoryNode):
    """This is a memory node for a reward (positive or negative)
    to the agent"""

    TABLE_COLUMNS = ["uuid", "value", "time"]
    TABLE = "Rewards"
    NODE_TYPE = "Reward"

    def __init__(self, agent_memory, memid: str):
        _, value, timestamp = agent_memory._db_read_one(
            "SELECT * FROM Rewards WHERE uuid=?", memid
        )
        self.value = value
        self.time = timestamp

    @classmethod
    def create(cls, agent_memory, reward_value: str) -> str:
        """Creates a new entry into the Rewards table
        
        Returns:
            string: memid of the entry
        
        Examples::
            >>> memory = AgentMemory()
            >>> reward_value = "positive"
            >>> create(memory, reward_value)
        """
        memid = cls.new(agent_memory)
        agent_memory.db_write(
            "INSERT INTO Rewards(uuid, value, time) VALUES (?,?,?)",
            memid,
            reward_value,
            agent_memory.get_time(),
        )
        return memid


NODELIST = NODELIST + [
    RewardNode,
    DanceNode,
    BlockTypeNode,
    SchematicNode,
    MobNode,
    ItemStackNode,
    MobTypeNode,
    InstSegNode,
    BlockObjectNode,
]  # noqa
