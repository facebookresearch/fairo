"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from collections import namedtuple
from droidlet.memory.craftassist.mc_memory import MCAgentMemory
from droidlet.memory.craftassist.mc_memory_nodes import (
    BlockObjectNode,
    MobNode,
    SchematicNode,
    InstSegNode,
    ItemStackNode,
    DanceNode,
    VoxelObjectNode,
)
from droidlet.memory.memory_nodes import PlayerNode, TripleNode
from droidlet.base_util import Pos, Look, Player
from droidlet.shared_data_struct.craftassist_shared_utils import Slot, Item, Mob, ItemStack


class BasicTest(unittest.TestCase):
    def test_basic_search_entityId(self):
        self.memory = MCAgentMemory()
        mob_memid = MobNode.create(self.memory, Mob(10, 65, Pos(1, 2, 3), Look(0, 0)))
        player_memid = PlayerNode.create(self.memory, Player(20, "xyz", Pos(1, 1, 1), Look(1, 1)))

        # Test getting entity with entityId
        _, entity_node = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE eid=10"
        )
        assert len(entity_node) == 1
        assert entity_node[0].pos == (1.0, 2.0, 3.0)
        _, entity_node = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE eid=20"
        )
        assert len(entity_node) == 1
        assert entity_node[0].pos == (1.0, 1.0, 1.0)

    def test_voxel_apis(self):
        self.memory = MCAgentMemory()
        bo_memid = BlockObjectNode.create(self.memory, [((1, 1, 1), (1, 2)), ((2, 2, 2), (2, 3))])
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 2
        # Test remove_voxel and test total number of blocks
        VoxelObjectNode.remove_voxel(self.memory, 1, 1, 1, "BlockObjects")
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 1

        # Test upsert_block (add a new block and assert that now there are two
        VoxelObjectNode.upsert_block(self.memory, ((3, 3, 3), (1, 1)), bo_memid, "BlockObjects")
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 2

    def test_block_objects_methods(self):
        self.memory = MCAgentMemory()
        bo_memid = BlockObjectNode.create(self.memory, [((1, 1, 1), (1, 2)), ((2, 2, 2), (2, 3))])

        # verify object is created, retrieve based on memid/uuid
        _, memnode = self.memory.basic_search(
            f"SELECT MEMORY FROM ReferenceObject WHERE uuid={bo_memid}"
        )
        assert len(memnode) == 1
        assert memnode[0].blocks == self.memory.get_mem_by_id(bo_memid).blocks
        # Test get_object_info_by_xyz byt cheking with BlockObject's mean xyz
        assert self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0] == bo_memid
        # Test get_object_info_by_xyz specialized for "BlockObjects"
        assert (
            self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0]
            == self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0]
        )
        assert self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0] == bo_memid
        # Test get_block_object_by_xyz -> same BlockObjectNode object as what we created before
        assert (
            self.memory.get_block_object_by_xyz((1, 1, 1)).blocks
            == self.memory.get_mem_by_id(bo_memid).blocks
        )
        # Test block object retrieval using id
        assert (
            self.memory.basic_search(
                f"SELECT MEMORY FROM ReferenceObject WHERE ref_type=BlockObjects AND uuid={bo_memid}"
            )[1][0].blocks
            == self.memory.get_mem_by_id(bo_memid).blocks
        )

    def test_inst_seg_node(self):
        self.memory = MCAgentMemory()
        inst_seg_memid = InstSegNode.create(
            self.memory, [(1, 0, 34), (1, 0, 35), (2, 0, 34), (3, 0, 34)], ["shiny", "bright"]
        )
        # Test get_instseg_object_ids_by_xyz
        assert self.memory.get_instseg_object_ids_by_xyz((1, 0, 34))[0][0] == inst_seg_memid
        assert self.memory.get_instseg_object_ids_by_xyz((2, 0, 34))[0][0] == inst_seg_memid
        assert self.memory.get_instseg_object_ids_by_xyz((3, 0, 34))[0][0] == inst_seg_memid

    def test_schematic_apis(self):
        self.memory = MCAgentMemory()
        schematic_memid = SchematicNode.create(
            self.memory, (((2, 0, 1), (1, 0)), ((2, 0, 2), (1, 0)), ((2, 0, 3), (2, 0)))
        )
        # test getting schematic by ID
        assert (
            self.memory.nodes[SchematicNode.NODE_TYPE](self.memory, schematic_memid).blocks
            == self.memory.get_mem_by_id(schematic_memid).blocks
        )
        # test getting schematic by its name
        self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=schematic_memid, pred_text="has_name", obj_text="house"
        )
        _, schematic_node = self.memory.basic_search(
            "SELECT MEMORY FROM Schematic WHERE has_name=house"
        )
        assert len(schematic_node) == 1
        assert schematic_node[0].blocks == self.memory.get_mem_by_id(schematic_memid).blocks
        bo1_memid = BlockObjectNode.create(self.memory, (((0, 0, 2), (1, 0)), ((0, 0, 3), (2, 0))))
        # Test convert_block_object_to_schematic
        assert (
            self.memory.nodes[SchematicNode.NODE_TYPE]
            .convert_block_object_to_schematic(self.memory, bo1_memid)
            .NODE_TYPE
            == "Schematic"
        )

    def test_set_mob_position(self):
        self.memory = MCAgentMemory()
        m = Mob(10, 65, Pos(1, 2, 3), Look(0, 0))
        assert self.memory.nodes[MobNode.NODE_TYPE].set_mob_position(self.memory, m).pos == (
            1.0,
            2.0,
            3.0,
        )

    def test_items_apis(self):
        self.memory = MCAgentMemory()
        low_level_data = {"bid_to_name": {(0, 0): "sand", (10, 4): "grass"}}
        sand_count = 2
        sand_bid = 0
        sand_meta = 0
        eid = 12
        item_memid = ItemStackNode.create(
            self.memory,
            ItemStack(Slot(sand_bid, sand_meta, sand_count), Pos(0, 0, 0), eid),
            low_level_data,
        )

        new_eid = 23
        # test update_item_stack_eid
        assert ItemStackNode.update_item_stack_eid(self.memory, item_memid, new_eid).eid == new_eid
        # test set_item_stack_position

        new_item = ItemStack(Slot(sand_bid, sand_meta, sand_count), Pos(2, 2, 2), new_eid)
        ItemStackNode.maybe_update_item_stack_position(self.memory, new_item)
        assert self.memory.get_mem_by_id(item_memid).pos == (2.0, 2.0, 2.0)

    def test_dance_api(self):
        self.memory = MCAgentMemory()

        def x():
            return 2

        self.memory.nodes[DanceNode.NODE_TYPE].create(
            self.memory, x, "generate_num_dance", ["generate_2", "dance_with_numbers"]
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, obj_text="generate_2"
                )
            )
            == 1
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, obj_text="dance_with_numbers"
                )
            )
            == 1
        )


if __name__ == "__main__":
    unittest.main()
