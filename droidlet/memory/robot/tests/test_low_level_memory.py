"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from droidlet.memory.robot.loco_memory import LocoAgentMemory
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode
from droidlet.base_util import Pos, Look, Player


class DO:
    def __init__(self, eid, label, properties, color, xyz, bounds, feature_repr=None):
        self.eid = eid
        self.label = label
        self.properties = properties
        self.color = color
        self.xyz = xyz
        self.bounds = bounds
        self.feature_repr = feature_repr

    def get_bounds(self):
        return self.bounds

    def get_xyz(self):
        return {"x": self.xyz[0], "y": self.xyz[1], "z": self.xyz[2]}


class BasicTest(unittest.TestCase):
    def test_player_apis(self):
        self.memory = LocoAgentMemory()
        player_list = [
            Player(20, "xyz", Pos(1, 1, 1), Look(1, 1)),
            Player(10, "abc", Pos(0, 0, 3), Look(0, 0))
        ]
        # test update_other_players
        self.memory.update_other_players(player_list)
        assert self.memory.get_player_by_name("xyz").pos == (1.0, 1.0, 1.0)
        assert self.memory.get_player_by_eid(10).name == "abc"

    def test_detected_object_apis(self):
        self.memory = LocoAgentMemory()
        d = DetectedObjectNode.create(self.memory, DO(
                  eid=33,
                  label="smaug",
                  properties = ["red_golden", "dragon", "lonely_mountain"],
                  color= "mauve",
                  xyz = [-0.4, -0.08, 0.0],
                  bounds = [0, 0, 0, 0, 0, 0]
              ))

        detected_object_mem_id = DetectedObjectNode.create(self.memory, d)
        # test get_detected_objects_tagged
        all_tags = ["red_golden", "dragon", "lonely_mountain", "mauve", "smaug"]
        for t in all_tags:
            assert len(self.memory.get_detected_objects_tagged(t)) == 1
            assert self.memory.get_detected_objects_tagged(t).pop() == detected_object_mem_id

    def test_dance_api(self):
        self.memory = LocoAgentMemory()

        def return_num():
            return 10
        self.memory.add_dance(return_num, "generate_num_10_dance", ["generate_num_10", "dance_with_numbers"])
        assert len(self.memory.get_triples(obj_text="generate_num_10")) == 1
        assert len(self.memory.get_triples(obj_text="dance_with_numbers")) == 1


if __name__ == "__main__":
    unittest.main()

