"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from collections import namedtuple
from droidlet.memory.craftassist.mc_memory import MCAgentMemory
from droidlet.memory.craftassist.mc_memory_nodes import BlockObjectNode, \
    MobNode, SchematicNode, InstSegNode, ItemStackNode, DanceNode
from droidlet.memory.memory_nodes import PlayerNode
from droidlet.base_util import Pos, Look, Player

Mob = namedtuple("Mob", "entityId, mobType, pos, look")
Item = namedtuple("item", "id, meta")
ItemStack = namedtuple("ItemStack", "entityId, item, pos")


class BasicTest(unittest.TestCase):

    def test_get_entity_by_id(self):
        self.memory = MCAgentMemory()
        mob_memid = MobNode.create(self.memory, Mob(10, 65, Pos(1, 2, 3), Look(0, 0)))
        player_memid = PlayerNode.create(self.memory, Player(20, "xyz", Pos(1, 1, 1), Look(1, 1)))

        # Test get_entity_by_eid with entityId
        assert self.memory.get_entity_by_eid(10).pos == (1.0, 2.0, 3.0)
        assert self.memory.get_entity_by_eid(20).pos == (1.0, 1.0, 1.0)

    def test_voxel_apis(self):
        self.memory = MCAgentMemory()
        bo_memid = BlockObjectNode.create(self.memory, [((1, 1, 1), (1, 2)), ((2, 2, 2), (2, 3))])
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 2
        # Test remove_voxel and test total number of blocks
        self.memory.remove_voxel(1, 1, 1, "BlockObjects")
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 1

        # Test upsert_block (add a new block and assert that now there are two
        self.memory.upsert_block(((3, 3, 3), (1, 1)), bo_memid, "BlockObjects")
        assert len(self.memory.get_mem_by_id(bo_memid).blocks) == 2

    def test_block_objects_methods(self):
        self.memory = MCAgentMemory()
        bo_memid = BlockObjectNode.create(self.memory, [((1, 1, 1), (1, 2)), ((2, 2, 2), (2, 3))])
        # Test get_object_by_id
        assert self.memory.get_mem_by_id(bo_memid).blocks == self.memory.get_object_by_id(bo_memid).blocks
        # Test get_object_info_by_xyz byt cheking with BlockObject's mean xyz
        assert self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0] == bo_memid
        # Test get_block_object_ids_by_xyz
        assert self.memory.get_block_object_ids_by_xyz((1, 1, 1))[0] == self.memory.get_object_info_by_xyz((1, 1, 1), "BlockObjects")[0]
        assert self.memory.get_block_object_ids_by_xyz((1, 1, 1))[0] == bo_memid
        # Test get_block_object_by_xyz -> same BlockObjectNode object as what we created before
        assert self.memory.get_block_object_by_xyz((1, 1, 1)).blocks == self.memory.get_mem_by_id(bo_memid).blocks
        # Test get_block_object_by_id
        assert self.memory.get_block_object_by_id(bo_memid).blocks == self.memory.get_mem_by_id(bo_memid).blocks

        # Test tag_block_object_from_schematic
        sch_memid = SchematicNode.create(self.memory, (((0, 0, 1), (1, 0)), ((0, 0, 2), (1, 0)), ((0, 0, 3), (2, 0))))
        bo1_memid = BlockObjectNode.create(self.memory, (((0, 0, 2), (1, 0)), ((0, 0, 3), (2, 0))))
        self.memory.tag_block_object_from_schematic(bo1_memid, sch_memid)
        assert len(self.memory.get_triples(pred_text="_from_schematic")) == 1

    def test_inst_seg_node(self):
        self.memory = MCAgentMemory()
        inst_seg_memid = InstSegNode.create(self.memory, [(1, 0, 34), (1, 0, 35), (2, 0, 34), (3, 0, 34)], ["shiny", "bright"])
        # Test get_instseg_object_ids_by_xyz
        assert self.memory.get_instseg_object_ids_by_xyz((1, 0, 34))[0][0] == inst_seg_memid
        assert self.memory.get_instseg_object_ids_by_xyz((2, 0, 34))[0][0] == inst_seg_memid
        assert self.memory.get_instseg_object_ids_by_xyz((3, 0, 34))[0][0] == inst_seg_memid

    def test_schematic_apis(self):
        self.memory = MCAgentMemory()
        schematic_memid = SchematicNode.create(self.memory, (((2, 0, 1), (1, 0)), ((2, 0, 2), (1, 0)), ((2, 0, 3), (2, 0))))
        # test get_schematic_by_id
        assert self.memory.get_schematic_by_id(schematic_memid).blocks == self.memory.get_mem_by_id(schematic_memid).blocks
        # test get_schematic_by_name
        self.memory.add_triple(subj=schematic_memid, pred_text="has_name", obj_text="house")
        assert self.memory.get_schematic_by_name("house").blocks == self.memory.get_mem_by_id(schematic_memid).blocks
        bo1_memid = BlockObjectNode.create(self.memory, (((0, 0, 2), (1, 0)), ((0, 0, 3), (2, 0))))
        # Test convert_block_object_to_schematic
        assert self.memory.convert_block_object_to_schematic(bo1_memid).NODE_TYPE == "Schematic"

    def test_set_mob_position(self):
        self.memory = MCAgentMemory()
        m = Mob(10, 65, Pos(1, 2, 3), Look(0, 0))
        assert self.memory.set_mob_position(m).pos == (1.0, 2.0, 3.0)

    def test_items_apis(self):
        self.memory = MCAgentMemory()
        low_level_data = {"block_data": {"bid_to_name": {
            (0, 0): "sand",
            (10, 4): "grass"
        }}}
        item_memid = ItemStackNode.create(self.memory, ItemStack(12, Item(0, 0), Pos(0, 0, 0)), low_level_data)
        # test update_item_stack_eid
        assert self.memory.update_item_stack_eid(item_memid, 23).eid == 23
        # test set_item_stack_position
        new_item = ItemStack(23, Item(0, 0), Pos(2, 2, 2))
        self.memory.set_item_stack_position(new_item)
        assert self.memory.get_mem_by_id(item_memid).pos == (2.0, 2.0, 2.0)
        # test get_all_item_stacks
        item_2_memid = ItemStackNode.create(self.memory, ItemStack(34, Item(10, 4), Pos(2, 3, 3)), low_level_data)
        assert len(self.memory.get_all_item_stacks()) == 2

    def test_dance_api(self):
        self.memory = MCAgentMemory()

        def x():
            return 2
        self.memory.add_dance(x, "generate_num_dance", ["generate_2", "dance_with_numbers"])
        assert len(self.memory.get_triples(obj_text="generate_2")) == 1
        assert len(self.memory.get_triples(obj_text="dance_with_numbers")) == 1


if __name__ == "__main__":
    unittest.main()
