"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import random
from typing import Optional, List
from droidlet.memory.sql_memory import AgentMemory
from droidlet.base_util import XYZ, Block, npy_to_blocks_list
from droidlet.memory.memory_nodes import (  # noqa
    TaskNode,
    PlayerNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    SetNode,
    ReferenceObjectNode,
)
from .mc_memory_nodes import (  # noqa
    DanceNode,
    VoxelObjectNode,
    BlockObjectNode,
    BlockTypeNode,
    MobNode,
    ItemStackNode,
    MobTypeNode,
    InstSegNode,
    SchematicNode,
    NODELIST,
)

PERCEPTION_RANGE = 64

# TODO: ship these schemas via setup.py and fix these directory references
SCHEMAS = [
    os.path.join(os.path.dirname(__file__), "..", "base_memory_schema.sql"),
    os.path.join(os.path.dirname(__file__), "mc_memory_schema.sql"),
]

SCHEMA = os.path.join(os.path.dirname(__file__), "memory_schema.sql")

THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

# TODO "snapshot" memory type  (giving a what mob/object/player looked like at a fixed timestamp)
# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class MCAgentMemory(AgentMemory):
    """Represents the memory for the agent in Minecraft"""

    def __init__(
        self,
        db_file=":memory:",
        db_log_path=None,
        schema_paths=SCHEMAS,
        load_minecraft_specs=True,
        load_block_types=True,
        preception_range=PERCEPTION_RANGE,
        agent_time=None,
        coordinate_transforms=None,
        agent_low_level_data={},
    ):
        super(MCAgentMemory, self).__init__(
            db_file=db_file,
            schema_paths=schema_paths,
            db_log_path=db_log_path,
            nodelist=NODELIST,
            agent_time=agent_time,
            coordinate_transforms=coordinate_transforms,
        )
        self.low_level_block_data = agent_low_level_data.get("block_data", {})
        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self._safe_pickle_saved_attrs = {}
        self.schematics = {}
        self.check_inside_perception = agent_low_level_data.get("check_inside", None)

        self._load_schematics(
            schematics=agent_low_level_data.get("schematics", {}),
            block_data=agent_low_level_data.get("block_data", {}),
            load_minecraft_specs=load_minecraft_specs,
        )
        self._load_block_types(
            block_data=agent_low_level_data.get("block_data", {}),
            color_data=agent_low_level_data.get("color_data", {}),
            block_property_data=agent_low_level_data.get("block_property_data", {}),
            load_block_types=load_block_types,
        )
        self._load_mob_types(
            mobs=agent_low_level_data.get("mobs", {}),
            mob_property_data=agent_low_level_data.get("mob_property_data", {}),
        )
        self.dances = {}
        self.perception_range = preception_range

    ###########################
    ### For Animate objects ###
    ###########################

    def get_entity_by_eid(self, eid) -> Optional["ReferenceObjectNode"]:
        """Find the entity node using the entity id.
        Returns:
            The memory node of entity with entity id
        """
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", eid)
        if r:
            return self.get_mem_by_id(r[0])
        else:
            return None

    ###############
    ### Voxels  ###
    ###############

    # FIXME: move these to VoxelObjectNode
    # count updates are done by hand to not need to count all voxels every time
    # use these functions, don't add/delete/modify voxels with raw sql
    def _update_voxel_count(self, memid, dn):
        """Update voxel count of a reference object with an amount
        equal to : dn"""
        c = self._db_read_one("SELECT voxel_count FROM ReferenceObjects WHERE uuid=?", memid)
        if c:
            count = c[0] + dn
            self.db_write("UPDATE ReferenceObjects SET voxel_count=? WHERE uuid=?", count, memid)
            return count
        else:
            return None

    def _update_voxel_mean(self, memid, count, loc):
        """update the x, y, z entries in ReferenceObjects
        to account for the removal or addition of a block.
        count should be the number of voxels *after* addition if >0
        and -count the number *after* removal if count < 0
        count should not be 0- handle that outside
        """
        old_loc = self._db_read_one("SELECT x, y, z  FROM ReferenceObjects WHERE uuid=?", memid)
        # TODO warn/error if no such memory?
        assert count != 0
        if old_loc:
            b = 1 / count
            if count > 0:
                a = (count - 1) / count
            else:
                a = (1 - count) / (-count)
            new_loc = (
                old_loc[0] * a + loc[0] * b,
                old_loc[1] * a + loc[1] * b,
                old_loc[2] * a + loc[2] * b,
            )
            self.db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?", *new_loc, memid
            )
            return new_loc

    def remove_voxel(self, x, y, z, ref_type):
        """Remove a voxel at (x, y, z) and of a given ref_type,
        and update the voxel count and mean as a result of the change"""
        memids = self._db_read_one(
            "SELECT uuid FROM VoxelObjects WHERE x=? and y=? and z=? and ref_type=?",
            x,
            y,
            z,
            ref_type,
        )
        if not memids:
            # TODO error/warning?
            return
        memid = memids[0]
        c = self._update_voxel_count(memid, -1)
        if c > 0:
            self._update_voxel_mean(memid, c, (x, y, z))
        self.db_write(
            "DELETE FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?", x, y, z, ref_type
        )

    def upsert_block(
        self,
        block: Block,
        memid: str,
        ref_type: str,
        player_placed: bool = False,
        agent_placed: bool = False,
        update: bool = True,  # if update is set to False, forces a write
    ):
        """This function upserts a block of ref_type in memory.
        Note:
        This functions only upserts to the same ref_type- if the voxel is
        occupied by a different ref_type it will insert a new ref object even if update is True"""

        ((x, y, z), (b, m)) = block
        old_memid = self._db_read_one(
            "SELECT uuid FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?",
            x,
            y,
            z,
            ref_type,
        )
        # add to voxel count
        new_count = self._update_voxel_count(memid, 1)
        assert new_count
        self._update_voxel_mean(memid, new_count, (x, y, z))
        if old_memid and update:
            if old_memid != memid:
                self.remove_voxel(x, y, z, ref_type)
                cmd = "INSERT INTO VoxelObjects (uuid, bid, meta, updated, player_placed, agent_placed, ref_type, x, y, z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            else:
                cmd = "UPDATE VoxelObjects SET uuid=?, bid=?, meta=?, updated=?, player_placed=?, agent_placed=? WHERE ref_type=? AND x=? AND y=? AND z=?"
        else:
            cmd = "INSERT INTO VoxelObjects (uuid, bid, meta, updated, player_placed, agent_placed, ref_type, x, y, z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self.db_write(
            cmd, memid, b, m, self.get_time(), player_placed, agent_placed, ref_type, x, y, z
        )

    def check_inside(self, mems):
        """ mems is a sequence of two ReferenceObjectNodes.
        this just wraps the heuristic perception check_inside method
        """
        return self.check_inside_perception(mems)

    ######################
    ###  BlockObjects  ###
    ######################

    def get_object_by_id(self, memid: str, table="BlockObjects") -> "VoxelObjectNode":
        """
        Returns:
            The memory node for the given memid
        """
        if table == "BlockObjects":
            return BlockObjectNode(self, memid)
        elif table == "InstSeg":
            return InstSegNode(self, memid)
        else:
            raise ValueError("Bad table={}".format(table))

    # and rename this
    def get_object_info_by_xyz(self, xyz: XYZ, ref_type: str, just_memid=True):
        """
        Returns:
            Memory node(s) at a given location and of a given ref_type
        """
        r = self._db_read(
            "SELECT DISTINCT(uuid), bid, meta FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?",
            *xyz,
            ref_type
        )
        if just_memid:
            return [memid for (memid, bid, meta) in r]
        else:
            return r

    # WARNING: these do not search archived/snapshotted block objects
    # TODO replace all these all through the codebase with generic counterparts
    def get_block_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        """Only get ids of memory node of type "BlockObjects" at (x, y, z)"""
        return self.get_object_info_by_xyz(xyz, "BlockObjects")

    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["VoxelObjectNode"]:
        """Get ids of memory node of type "BlockObjects" or "VoxelObjectNode"
        at (x, y, z)"""
        memids = self.get_block_object_ids_by_xyz(xyz)
        if len(memids) == 0:
            return None
        return self.get_block_object_by_id(memids[0])

    def get_block_object_by_id(self, memid: str) -> "VoxelObjectNode":
        return self.get_object_by_id(memid, "BlockObjects")

    def tag_block_object_from_schematic(self, block_object_memid: str, schematic_memid: str):
        """Tag a block object that came from a schematic"""
        self.add_triple(subj=block_object_memid, pred_text="_from_schematic", obj=schematic_memid)

    #####################
    ### InstSegObject ###
    #####################

    def get_instseg_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        """Get ids of memory nodes of ref_type: "inst_seg" using their
        location"""
        r = self._db_read(
            'SELECT DISTINCT(uuid) FROM VoxelObjects WHERE ref_type="inst_seg" AND x=? AND y=? AND z=?',
            *xyz
        )
        return r

    ####################
    ###  Schematics  ###
    ####################

    def get_schematic_by_id(self, memid: str) -> "SchematicNode":
        """Get the Schematic type memory node using id"""
        return SchematicNode(self, memid)

    def _get_schematic_by_property_name(self, name, table_name) -> Optional["SchematicNode"]:
        """Get the Schematic type memory node using name"""
        r = self._db_read(
            """
                    SELECT {}.type_name
                    FROM {} INNER JOIN Triples as T ON T.subj={}.uuid
                    WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""".format(
                table_name, table_name, table_name
            ),
            name,
        )
        if not r:
            return None

        result = []  # noqa
        for e in r:
            schematic_name = e[0]
            schematics = self._db_read(
                """
                    SELECT Schematics.uuid
                    FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                    WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""",
                schematic_name,
            )
            if schematics:
                result.extend(schematics)
        if result:
            return self.get_schematic_by_id(random.choice(result)[0])
        else:
            return None

    def get_schematic_by_name(self, name: str) -> Optional["SchematicNode"]:
        """Get the id of Schematic type memory node using name"""
        r = self._db_read(
            """
                SELECT Schematics.uuid
                FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""",
            name,
        )
        if r:  # if multiple exist, then randomly select one
            return self.get_schematic_by_id(random.choice(r)[0])
        # if no schematic with exact matched name exists, search for a schematic
        # with matched property name instead
        else:
            return self._get_schematic_by_property_name(name, "BlockTypes")

    def convert_block_object_to_schematic(self, block_object_memid: str) -> "SchematicNode":
        """Save a BlockObject as a Schematic node along with the link"""
        r = self._db_read_one(
            'SELECT subj FROM Triples WHERE pred_text="_source_block_object" AND obj=?',
            block_object_memid,
        )
        if r:
            # previously converted; return old schematic
            return self.get_schematic_by_id(r[0])

        else:
            # get up to date BlockObject
            block_object = self.get_block_object_by_id(block_object_memid)

            # create schematic
            memid = SchematicNode.create(self, list(block_object.blocks.items()))

            # add triple linking the object to the schematic
            self.add_triple(subj=memid, pred_text="_source_block_object", obj=block_object.memid)

            return self.get_schematic_by_id(memid)

    def _load_schematics(self, schematics, block_data, load_minecraft_specs=True):
        """Load all Minecraft schematics into agent memory"""
        if load_minecraft_specs:
            for premem in schematics:
                npy = premem["schematic"]

                # lazy loading, only store memid in db, ((0, 0, 0), (0, 0)) as a placeholder
                memid = SchematicNode.create(self, [((0, 0, 0), (0, 0))])
                self.schematics[memid] = npy_to_blocks_list(npy)

                if premem.get("name"):
                    for n in premem["name"]:
                        self.add_triple(subj=memid, pred_text="has_name", obj_text=n)
                        self.add_triple(subj=memid, pred_text="has_tag", obj_text=n)
                if premem.get("tags"):
                    for t in premem["tags"]:
                        self.add_triple(subj=memid, pred_text="has_tag", obj_text=t)

        # load single blocks as schematics
        bid_to_name = block_data.get("bid_to_name", {})
        for (d, m), name in bid_to_name.items():
            if d >= 256:
                continue
            memid = SchematicNode.create(self, [((0, 0, 0), (d, m))])
            self.add_triple(subj=memid, pred_text="has_name", obj_text=name)
            if "block" in name:
                self.add_triple(
                    subj=memid, pred_text="has_name", obj_text=name.strip("block").strip()
                )
            # tag single blocks with 'block'
            self.add_triple(subj=memid, pred_text="has_name", obj_text="block")

    def _load_block_types(
        self,
        block_data,
        color_data,
        block_property_data,
        load_block_types=True,
        load_color=True,
        load_block_property=True,
        simple_color=False,
        load_material=True,
    ):
        """Load all block types into agent memory"""
        if not load_block_types:
            return

        if simple_color:
            name_to_colors = color_data.get("name_to_simple_colors", {})
        else:
            name_to_colors = color_data.get("name_to_colors", {})

        block_name_to_properties = block_property_data.get("name_to_properties", {})

        bid_to_name = block_data.get("bid_to_name", {})

        for (b, m), type_name in bid_to_name.items():
            if b >= 256:
                continue
            memid = BlockTypeNode.create(self, type_name, (b, m))
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            if "block" in type_name:
                self.add_triple(
                    subj=memid, pred_text="has_name", obj_text=type_name.strip("block").strip()
                )

            if load_color:
                if name_to_colors.get(type_name) is not None:
                    for color in name_to_colors[type_name]:
                        self.add_triple(subj=memid, pred_text="has_colour", obj_text=color)

            if load_block_property:
                if block_name_to_properties.get(type_name) is not None:
                    for property in block_name_to_properties[type_name]:
                        self.add_triple(subj_text=memid, pred_text="has_name", obj_text=property)

    def _load_mob_types(self, mobs, mob_property_data, load_mob_types=True):
        """Load all mob types into agent memory"""
        if not load_mob_types:
            return

        mob_name_to_properties = mob_property_data.get("name_to_properties", {})
        for (name, m) in mobs.items():
            type_name = "spawn " + name

            # load single mob as schematics
            memid = SchematicNode.create(self, [((0, 0, 0), (383, m))])
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            self.tag(memid, "_spawn")
            self.tag(memid, name)
            if "block" in name:
                self.tag(memid, name.strip("block").strip())

            # then load properties
            memid = MobTypeNode.create(self, type_name, (383, m))
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            if mob_name_to_properties.get(type_name) is not None:
                for prop in mob_name_to_properties[type_name]:
                    self.tag(memid, prop)

    ##############
    ###  Mobs  ###
    ##############

    def set_mob_position(self, mob) -> "MobNode":
        """Update the position of mob in memory"""
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", mob.entityId)
        if r:
            self.db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=?, yaw=?, pitch=? WHERE eid=?",
                mob.pos.x,
                mob.pos.y,
                mob.pos.z,
                mob.look.yaw,
                mob.look.pitch,
                mob.entityId,
            )
            (memid,) = r
        else:
            memid = MobNode.create(self, mob)
        return self.get_mem_by_id(memid)

    ####################
    ###  ItemStacks  ###
    ####################

    def update_item_stack_eid(self, memid, eid) -> "ItemStackNode":
        """Update ItemStack in memory and return the corresponding node
        Returns:
            ItemStackNode
        """
        r = self._db_read_one("SELECT * FROM ReferenceObjects WHERE uuid=?", memid)
        if r:
            self.db_write("UPDATE ReferenceObjects SET eid=? WHERE uuid=?", eid, memid)
        return self.get_mem_by_id(memid)

    def set_item_stack_position(self, item_stack) -> "ItemStackNode":
        """If the node exists, update the position of item stack in memory
        else create a new node.
        Returns :
            Updated or new ItemStackNode
        """
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", item_stack.entityId)
        if r:
            self.db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE eid=?",
                item_stack.pos.x,
                item_stack.pos.y,
                item_stack.pos.z,
                item_stack.entityId,
            )
            (memid,) = r
        else:
            memid = ItemStackNode.create(self, item_stack, self.low_level_block_data)
        return self.get_mem_by_id(memid)

    def get_all_item_stacks(self):
        """Get all nodes that are of type "item_stack" """
        r = self._db_read("SELECT uuid, eid FROM ReferenceObjects WHERE ref_type=?", "item_stack")
        return r

    ###############
    ###  Dances  ##
    ###############

    def add_dance(self, dance_fn, name=None, tags=[]):
        """Add a dance movement to memory"""
        # a dance is movement determined as a sequence of steps, rather than by its destination
        return DanceNode.create(self, dance_fn, name=name, tags=tags)
