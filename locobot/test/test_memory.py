"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
import os
import unittest
from locobot.agent.loco_memory import LocoAgentMemory
from locobot.agent.loco_memory_nodes import DetectedObjectNode, HumanPoseNode
from utils import get_fake_detection, get_fake_humanpose


class MemoryTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MemoryTests, self).__init__(*args, **kwargs)
        self.memory = LocoAgentMemory()

    def test_detected_object_node_creation(self):
        d = get_fake_detection(
            "smaug", ["red_golden", "dragon", "lonely_mountain"], [-0.4, -0.08, 0.0]
        )
        DetectedObjectNode.create(self.memory, d)

        # Check that class_label, properties are saved as tags
        tags_to_check = ["red_golden", "dragon", "lonely_mountain", "smaug"]
        for x in tags_to_check:
            t = self.memory.get_detected_objects_tagged(x)
            self.assertEqual(len(t), 1)

    def test_humanpose_node_creation(self):
        h = get_fake_humanpose()
        HumanPoseNode.create(self.memory, h)

        # Check that _human_pose is a tag as per implementation
        tags_to_check = ["_human_pose"]
        for x in tags_to_check:
            t = self.memory.get_detected_objects_tagged(x)
            self.assertEqual(len(t), 1)


if __name__ == "__main__":
    unittest.main()
