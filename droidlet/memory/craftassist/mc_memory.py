"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import copy
import os
import random
import numpy as np
from collections import namedtuple
from typing import Optional, List
from droidlet.memory.sql_memory import AgentMemory, DEFAULT_PIXELS_PER_UNIT
from droidlet.base_util import Pos
from droidlet.shared_data_struct.craftassist_shared_utils import ItemStack
from droidlet.base_util import diag_adjacent, IDM, XYZ, Block, npy_to_blocks_list
from droidlet.memory.memory_nodes import (  # noqa
    TaskNode,
    SelfNode,
    PlayerNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    SetNode,
    ReferenceObjectNode,
    AttentionNode,
    TripleNode,
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
        perception_range=PERCEPTION_RANGE,
        agent_time=None,
        coordinate_transforms=None,
        agent_low_level_data={},
        place_field_pixels_per_unit=DEFAULT_PIXELS_PER_UNIT,
        copy_from_backup=None,
    ):
        super(MCAgentMemory, self).__init__(
            db_file=db_file,
            schema_paths=schema_paths,
            db_log_path=db_log_path,
            nodelist=NODELIST,
            agent_time=agent_time,
            coordinate_transforms=coordinate_transforms,
            place_field_pixels_per_unit=place_field_pixels_per_unit,
        )
        self.low_level_block_data = agent_low_level_data.get("block_data", {})
        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self._safe_pickle_saved_attrs = {}
        self.schematics = {}
        self.check_inside_perception = agent_low_level_data.get("check_inside", None)

        self.dances = {}
        self.perception_range = perception_range

        if copy_from_backup is not None:
            copy_from_backup.backup(self.db)
            self.make_self_mem()
        else:
            self.nodes[SchematicNode.NODE_TYPE]._load_schematics(
                self,
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

    ############################################
    ### Update world with perception updates ###
    ############################################

    def update(self, perception_output: namedtuple = None, areas_to_perceive: List = []):
        """
        Updates the world with updates from agent's perception module.

        Args:
            perception_output: namedtuple with attributes-
                mob: All mobs in perception range.
                agent_pickable_items: Dict containing - items in agent's perception that can be picked up
                    and all items that can be picked up by the agent.
                agent_attributes: Agent's player attributes including
                other_player_list: List of other in-game players
                changed_block_attributes: marked attributes (interesting, player_placed, agent_placed)
                    of changed blocks
                in_perceive_area : blockobjects, holes and blocks in the area agent will be running perception in
                near_agent: block objects, holes and blocks near the agent
                labeled_blocks: labels and resulting locations from semantic segmentation model
                dashboard_edits: Dict of ReferenceObject memid's, edits made to attributes from dashboard
                dashboard_groups: Dict of ReferenceObject memid's, tags annotated by user in dashboard

        :return:
        updated_areas_to_perceive: list of (xyz, idm) representing the area agent should perceive
        """
        if not perception_output:
            return areas_to_perceive
        self_node = self.get_mem_by_id(self.self_memid)
        output = {}
        updated_areas_to_perceive = areas_to_perceive

        """Perform update to memory with input from low_level perception module"""
        # 1. Handle all mobs in agent's perception range
        if perception_output.mobs:
            map_changes = []
            for mob in perception_output.mobs:
                mob_memid = self.nodes[MobNode.NODE_TYPE].set_mob_position(self, mob)
                mp = (mob.pos.x, mob.pos.y, mob.pos.z)
                map_changes.append(
                    {"pos": mp, "is_obstacle": False, "memid": mob_memid, "is_move": True}
                )
            # FIXME track these semi-automatically...
            self.place_field.update_map(map_changes)

        # 2. Handle all items that the agent can pick up in-game
        holder_eids = {}
        # FIXME: deal with far away things better
        for eid, item_stack_info in perception_output.agent_pickable_items.items():
            struct, holder_eid, tags = item_stack_info
            holder_eids[struct.entityId] = holder_eid
            if (
                np.linalg.norm(np.array(self_node.pos) - np.array(struct.pos))
                < self.perception_range
            ):
                node = ItemStackNode.maybe_update_item_stack_position(self, struct)
                if not node:
                    memid = ItemStackNode.create(self, struct, self.low_level_block_data)
                else:
                    memid = node.memid
                TripleNode.untag(self, memid, "_possibly_stale_location")
                # TODO: remove stale triples?
                for pred_text, obj_text in tags:
                    TripleNode.create(self, subj=memid, pred_text=pred_text, obj_text=obj_text)

        # cuberite/mc does not return item_stacks in agent's or others inventory.
        # we do the best we can with these, FIXME
        # not removing any old items, FIXME
        all_item_stacks = self._db_read(
            "SELECT uuid, eid FROM ReferenceObjects WHERE ref_type=?", "item_stack"
        )
        for memid, eid in all_item_stacks:
            holder_eid = holder_eids.get(eid)
            if holder_eid is not None:
                old_triples = self._db_read(
                    "SELECT uuid FROM Triples WHERE subj=? AND pred_text=?", memid, "held_by"
                )
                for uuid in old_triples:
                    self.forget(uuid)
                if holder_eid == -1:
                    node = self.get_mem_by_id(memid)
                    TripleNode.tag(self, memid, "_on_ground")
                    TripleNode.untag(self, memid, "_in_inventory")
                    TripleNode.untag(self, memid, "_in_others_inventory")
                else:
                    r = self._db_read_one(
                        "SELECT uuid FROM ReferenceObjects WHERE eid=?", holder_eid
                    )
                    if not r:
                        raise Exception(
                            "holder eid from perception given as {} but entity not found in ReferenceObjects".format(
                                holder_eid
                            )
                        )
                    old_triples = self._db_read(
                        "SELECT uuid FROM Triples WHERE subj=? AND pred_text=?", memid, "held_by"
                    )
                    for uuid in old_triples:
                        self.forget(uuid)
                    TripleNode.create(self, subj=memid, pred_text="held_by", obj=r[0])
                    TripleNode.untag(self, memid, "_on_ground")
                    if holder_eid == self_node.eid:
                        TripleNode.tag(self, memid, "_in_inventory")
                    else:
                        TripleNode.tag(self, memid, "_in_others_inventory")
            else:
                node = self.get_mem_by_id(memid)
                # we are in cuberite, and an item is held by another entity or has disappeared
                # in any case, we can't track its location
                TripleNode.tag(self, memid, "_possibly_stale_location")
        held_memids = TripleNode.get_memids_by_tag(self, "_in_inventory")
        for memid in held_memids:
            eid = self._db_read_one("SELECT eid FROM ReferenceObjects WHERE uuid=?", memid)[0]
            struct = ItemStack(None, Pos(*self_node.pos), eid, "")
            ItemStackNode.maybe_update_item_stack_position(self, struct)

        # 3. Update agent's current position and attributes in memory
        if perception_output.agent_attributes:
            agent_player = perception_output.agent_attributes
            cmd = (
                "UPDATE ReferenceObjects SET eid=?, name=?, x=?,  y=?, z=?, pitch=?, yaw=? WHERE "
            )
            cmd = cmd + "uuid=?"
            self.db_write(
                cmd,
                agent_player.entityId,
                agent_player.name,
                agent_player.pos.x,
                agent_player.pos.y,
                agent_player.pos.z,
                agent_player.look.pitch,
                agent_player.look.yaw,
                self.self_memid,
            )
            ap = (agent_player.pos.x, agent_player.pos.y, agent_player.pos.z)
            self.place_field.update_map(
                [{"pos": ap, "is_obstacle": True, "memid": self.self_memid, "is_move": True}]
            )

        # 4. Update other in-game players in agent's memory
        if perception_output.other_player_list:
            player_list = perception_output.other_player_list
            for player, location in player_list:
                mem = self.nodes[PlayerNode.NODE_TYPE].get_player_by_eid(self, player.entityId)
                if mem is None:
                    memid = self.nodes[PlayerNode.NODE_TYPE].create(self, player)
                else:
                    memid = mem.memid
                cmd = "UPDATE ReferenceObjects SET eid=?, name=?, x=?,  y=?, z=?, pitch=?, yaw=? WHERE "
                cmd = cmd + "uuid=?"
                self.db_write(
                    cmd,
                    player.entityId,
                    player.name,
                    player.pos.x,
                    player.pos.y,
                    player.pos.z,
                    player.look.pitch,
                    player.look.yaw,
                    memid,
                )
                pp = (player.pos.x, player.pos.y, player.pos.z)
                self.place_field.update_map(
                    [{"pos": pp, "is_obstacle": True, "memid": memid, "is_move": True}]
                )
                memids = self._db_read_one(
                    'SELECT uuid FROM ReferenceObjects WHERE ref_type="attention" AND type_name=?',
                    player.entityId,
                )
                if memids:
                    self.db_write(
                        "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?",
                        location[0],
                        location[1],
                        location[2],
                        memids[0],
                    )
                else:
                    AttentionNode.create(self, location, attender=player.entityId)

        # 5. Update the state of the world when a block is changed.
        if perception_output.changed_block_attributes:
            for (xyz, idm) in perception_output.changed_block_attributes:
                # 5.1 Update old instance segmentation if needed
                self.maybe_remove_inst_seg(xyz)

                # 5.2 Update agent's memory with blocks that have been destroyed.
                updated_areas_to_perceive = self.maybe_remove_block_from_memory(
                    xyz, idm, areas_to_perceive
                )

                # 5.3 Update blocks in memory when any change in the environment is caused either by agent or player
                (
                    interesting,
                    player_placed,
                    agent_placed,
                ) = perception_output.changed_block_attributes[(xyz, idm)]
                self.maybe_add_block_to_memory(interesting, player_placed, agent_placed, xyz, idm)

        """Now perform update to memory with input from heuristic perception module"""
        # 1. Process everything in area to attend for perception
        if perception_output.in_perceive_area:
            # 1.1 Add colors of all block objects
            if perception_output.in_perceive_area["block_object_attributes"]:
                for block_object_attr in perception_output.in_perceive_area[
                    "block_object_attributes"
                ]:
                    block_object, color_tags = block_object_attr
                    memid = BlockObjectNode.create(self, block_object)
                    for color_tag in list(set(color_tags)):
                        TripleNode.create(
                            self, subj=memid, pred_text="has_colour", obj_text=color_tag
                        )
            # 1.2 Update all holes with their block type in memory
            if perception_output.in_perceive_area["holes"]:
                self.add_holes_to_mem(perception_output.in_perceive_area["holes"])
            # 1.3 Update tags of air-touching blocks
            if "airtouching_blocks" in perception_output.in_perceive_area:
                for c, tags in perception_output.in_perceive_area["airtouching_blocks"]:
                    InstSegNode.create(self, c, tags=tags)
        # 2. Process everything near agent's current position
        if perception_output.near_agent:
            # 2.1 Add colors of all block objects
            if perception_output.near_agent["block_object_attributes"]:
                for block_object_attr in perception_output.near_agent["block_object_attributes"]:
                    block_object, color_tags = block_object_attr
                    memid = BlockObjectNode.create(self, block_object)
                    for color_tag in list(set(color_tags)):
                        TripleNode.create(
                            self, subj=memid, pred_text="has_colour", obj_text=color_tag
                        )
            # 2.2 Update all holes with their block type in memory
            if perception_output.near_agent["holes"]:
                self.add_holes_to_mem(perception_output.near_agent["holes"])
            # 2.3 Update tags of air-touching blocks
            if "airtouching_blocks" in perception_output.near_agent:
                for c, tags in perception_output.near_agent["airtouching_blocks"]:
                    InstSegNode.create(self, c, tags=tags)

        """Update the memory with labeled blocks from SubComponent classifier"""
        if perception_output.labeled_blocks:
            for label, locations in perception_output.labeled_blocks.items():
                InstSegNode.create(self, locations, [label])

        """Update the memory with holes"""
        if perception_output.holes:
            hole_memories = self.add_holes_to_mem(perception_output.holes)
            output["holes"] = hole_memories

        """Now perform update to memory with input from manual edits perception module"""
        if perception_output.dashboard_edits:
            self.make_manual_edits(perception_output.dashboard_edits)
        if perception_output.dashboard_groups:
            self.make_dashboard_groups(perception_output.dashboard_groups)

        output["areas_to_perceive"] = updated_areas_to_perceive
        return output

    def maybe_add_block_to_memory(self, interesting, player_placed, agent_placed, xyz, idm):
        if not interesting:
            return

        adjacent = [
            self.get_object_info_by_xyz(a, "BlockObjects", just_memid=False)
            for a in diag_adjacent(xyz)
        ]
        if idm[0] == 0:
            # block removed / air block added
            adjacent_memids = [a[0][0] for a in adjacent if len(a) > 0 and a[0][1] == 0]
        else:
            # normal block added
            adjacent_memids = [a[0][0] for a in adjacent if len(a) > 0 and a[0][1] > 0]
        adjacent_memids = list(set(adjacent_memids))
        if len(adjacent_memids) == 0:
            # new block object
            memid = BlockObjectNode.create(self, [(xyz, idm)])
            self.place_field.update_map([{"pos": xyz, "is_obstacle": True, "memid": memid}])
        elif len(adjacent_memids) == 1:
            # update block object
            memid = adjacent_memids[0]
            VoxelObjectNode.upsert_block(
                self, (xyz, idm), memid, "BlockObjects", player_placed, agent_placed
            )
            self.place_field.update_map([{"pos": xyz, "is_obstacle": True, "memid": memid}])
            self.set_memory_updated_time(memid)
            self.set_memory_attended_time(memid)
        else:
            chosen_memid = adjacent_memids[0]
            self.set_memory_updated_time(chosen_memid)
            self.set_memory_attended_time(chosen_memid)

            # merge tags
            where = " OR ".join(["subj=?"] * len(adjacent_memids))
            self.db_write(
                "UPDATE Triples SET subj=? WHERE " + where, chosen_memid, *adjacent_memids
            )

            # merge multiple block objects (will delete old ones)
            where = " OR ".join(["uuid=?"] * len(adjacent_memids))
            cmd = "UPDATE VoxelObjects SET uuid=? WHERE "
            self.db_write(cmd + where, chosen_memid, *adjacent_memids)

            # insert new block
            VoxelObjectNode.upsert_block(
                self, (xyz, idm), chosen_memid, "BlockObjects", player_placed, agent_placed
            )

    def add_holes_to_mem(self, holes):
        """
        Adds the list of holes to memory and return hole memories.
        """
        hole_memories = []
        for hole in holes:
            memid = InstSegNode.create(self, hole[0], tags=["hole", "pit", "mine"])
            try:
                fill_block_name = self.low_level_block_data["bid_to_name"][hole[1]]
            except:
                idm = (hole[1][0], 0)
                fill_block_name = self.low_level_block_data["bid_to_name"].get(idm)
            if fill_block_name:
                query = "SELECT MEMORY FROM BlockType WHERE has_name={}".format(fill_block_name)
                _, fill_block_mems = self.basic_search(query)
                fill_block_memid = fill_block_mems[0].memid
                TripleNode.create(
                    self, subj=memid, pred_text="has_fill_type", obj=fill_block_memid
                )
            hole_memories.append(self.get_mem_by_id(memid))
        return hole_memories

    def maybe_remove_block_from_memory(self, xyz: XYZ, idm: IDM, areas_to_perceive):
        """Update agent's memory with blocks that have been destroyed."""
        tables = ["BlockObjects"]
        local_areas_to_perceive = copy.deepcopy(areas_to_perceive)
        for table in tables:
            info = self.get_object_info_by_xyz(xyz, table, just_memid=False)
            if not info or len(info) == 0:
                continue
            assert len(info) == 1
            memid, b, m = info[0]
            delete = (b == 0 and idm[0] > 0) or (b > 0 and idm[0] == 0)
            if delete:
                VoxelObjectNode.remove_voxel(self, *xyz, table)
                # check if the whole column is removed:
                # FIXME, eventually want y slices
                r = self._db_read(
                    "SELECT uuid FROM VoxelObjects WHERE x=? AND z=? and ref_type=?",
                    xyz[0],
                    xyz[2],
                    tables[0],
                )
                if len(r) == 0:
                    self.place_field.update_map([{"pos": xyz, "is_delete": True}])
                local_areas_to_perceive.append((xyz, 3))
        return local_areas_to_perceive

    def maybe_remove_inst_seg(self, xyz: XYZ):
        """if the block is changed, the old instance segmentation
        is no longer considered valid"""
        # get all associated instseg nodes
        # FIXME make this into a basic search
        inst_seg_memids = self.get_instseg_object_ids_by_xyz(xyz)
        if inst_seg_memids:
            # delete the InstSeg, they are ephemeral and should be recomputed
            # TODO/FIXME  more refined approach: if a block changes
            # ask the models to recompute.  if the tags are the same, keep it
            for i in inst_seg_memids:
                self.forget(i[0])

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

    def check_inside(self, mems):
        """mems is a sequence of two ReferenceObjectNodes.
        this just wraps the heuristic perception check_inside method
        """
        return self.check_inside_perception(mems)

    ######################
    ###  BlockObjects  ###
    ######################

    # and rename this
    # FIXME can not be simply deleted and replaced with basic_search
    # since block-objects are indexed by their avg-xyz not per-xyz
    def get_object_info_by_xyz(self, xyz: XYZ, ref_type: str, just_memid=True):
        """
        Returns:
            Memory node(s) at a given location and of a given ref_type
        """
        r = self._db_read(
            "SELECT DISTINCT(uuid), bid, meta FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?",
            *xyz,
            ref_type,
        )
        if just_memid:
            return [memid for (memid, bid, meta) in r]
        else:
            return r

    # WARNING: these do not search archived/snapshotted block objects
    # TODO replace all these all through the codebase with generic counterparts
    # FIXME can not be simply replaced by basic_search, not sure how to use
    # _db_read to access objects instead of uuid's
    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["VoxelObjectNode"]:
        """Get ids of memory node of type "BlockObjects" or "VoxelObjectNode"
        at (x, y, z)"""
        memids = self.get_object_info_by_xyz(xyz, "BlockObjects")
        if len(memids) == 0:
            return None
        return self.basic_search(
            f"SELECT MEMORY FROM ReferenceObject WHERE ref_type=BlockObjects AND uuid={memids[0]}"
        )[1][0]

    #####################
    ### InstSegObject ###
    #####################

    # TODO can not be directly replaced with basic-search, since it searches
    # over ReferenceObjects which are indexed by avg-xyz not per-xyz
    def get_instseg_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        """Get ids of memory nodes of ref_type: "inst_seg" using their
        location"""
        r = self._db_read(
            'SELECT DISTINCT(uuid) FROM VoxelObjects WHERE ref_type="inst_seg" AND x=? AND y=? AND z=?',
            *xyz,
        )
        return r

    ########################
    ###  DashboardEdits  ###
    ########################
    def make_manual_edits(self, edits):
        for memid in edits.keys():
            toEdit = {
                attr: val for attr, val in edits[memid].items() if attr not in ("location", "pos")
            }
            if toEdit:
                cmd = (
                    "UPDATE ReferenceObjects SET " + "=?, ".join(toEdit.keys()) + "=? WHERE uuid=?"
                )
                self.db_write(cmd, *toEdit.values(), memid)

            # spatial data is iterable, needs to be handled differently
            if "pos" in edits[memid].keys():
                newPos = edits[memid]["pos"]
                assert len(newPos) == 3
                cmd = "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?"
                self.db_write(cmd, newPos[0], newPos[1], newPos[2], memid)
            elif "location" in edits[memid].keys():
                newPos = edits[memid]["location"]
                assert len(newPos) == 3
                cmd = "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?"
                self.db_write(cmd, newPos[0], newPos[1], newPos[2], memid)

    def make_dashboard_groups(self, groups):
        for group, memids in groups.items():
            for memid in memids:
                TripleNode.create(self, subj=memid, pred_text="is_a", obj_text=group)

    ####################
    ###  Schematics  ###
    ####################

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
            TripleNode.create(self, subj=memid, pred_text="has_name", obj_text=type_name)
            if "block" in type_name:
                TripleNode.create(
                    self,
                    subj=memid,
                    pred_text="has_name",
                    obj_text=type_name.strip("block").strip(),
                )

            if load_color:
                if name_to_colors.get(type_name) is not None:
                    for color in name_to_colors[type_name]:
                        TripleNode.create(self, subj=memid, pred_text="has_colour", obj_text=color)

            if load_block_property:
                if block_name_to_properties.get(type_name) is not None:
                    for property in block_name_to_properties[type_name]:
                        TripleNode.create(
                            self, subj_text=memid, pred_text="has_name", obj_text=property
                        )

    def _load_mob_types(self, mobs, mob_property_data, load_mob_types=True):
        """Load all mob types into agent memory"""
        if not load_mob_types:
            return

        mob_name_to_properties = mob_property_data.get("name_to_properties", {})
        for (name, m) in mobs.items():
            type_name = "spawn " + name

            # load single mob as schematics
            memid = SchematicNode.create(self, [((0, 0, 0), (383, m))])
            TripleNode.create(self, subj=memid, pred_text="has_name", obj_text=type_name)
            TripleNode.tag(self, memid, "_spawn")
            TripleNode.tag(self, memid, name)
            if "block" in name:
                TripleNode.tag(self, memid, name.strip("block").strip())

            # then load properties
            memid = MobTypeNode.create(self, type_name, (383, m))
            TripleNode.create(self, subj=memid, pred_text="has_name", obj_text=type_name)
            if mob_name_to_properties.get(type_name) is not None:
                for prop in mob_name_to_properties[type_name]:
                    TripleNode.tag(self, memid, prop)
