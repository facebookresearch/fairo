"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
import os

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(BASE_AGENT_ROOT)

import unittest
import logging
from timeit import Timer
from unittest.mock import MagicMock, Mock
from locobot.agent.perception import (
    InputHandler,
    DetectionHandler,
    FaceRecognitionHandler,
    ObjectDeduplicationHandler,
    MemoryHandler,
    SlowPerception,
    Detection,
    RGBDepth,
    Human,
)
from locobot.agent.loco_memory import LocoAgentMemory
from locobot.agent.loco_memory_nodes import DetectedObjectNode, HumanPoseNode
from locobot.agent.locobot_mover import LoCoBotMover
import cv2
import torch
from PIL import Image
from utils import get_fake_rgbd, get_fake_detection, get_fake_humanpose

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IP = "127.0.0.1"
if os.getenv("LOCOBOT_IP"):
    IP = os.getenv("LOCOBOT_IP")

PERCEPTION_MODELS_DIR = "locobot/agent/models/perception"
OFFICE_IMG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "test_assets/perception_handlers",
    "office_chair.jpg",
)
GROUP_IMG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "test_assets/perception_handlers",
    "obama_trump.jpg",
)
FACES_IDS_DIR = os.path.join(os.path.dirname(__file__), "test_assets/perception_handlers/faces")

logging.getLogger().setLevel(logging.INFO)


class PerceiveTimeTest(unittest.TestCase):
    def setUp(self) -> None:
        m_agent = MagicMock()
        m_agent.mover = LoCoBotMover(ip=IP, backend="habitat")
        self.input_handler = InputHandler(m_agent, read_from_camera=True)
        self.perception = SlowPerception(PERCEPTION_MODELS_DIR)

    def test_time(self):
        rgb_depth = self.input_handler.handle()
        t = Timer(lambda: self.perception.perceive(rgb_depth))
        logging.info("SlowPerception runtime {} s".format(t.timeit(number=1)))


class InputHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        m_agent = MagicMock()
        m_agent.mover = LoCoBotMover(ip=IP, backend="habitat")
        self.input_handler = InputHandler(m_agent, read_from_camera=True)

    def test_handler(self):
        # check the type of the returned object from input handler function
        rgb_depth = self.input_handler.handle()
        # check if the returned object can return the actual image
        get_pillow_method = getattr(rgb_depth, "get_pillow_image")
        self.assertEqual(callable(get_pillow_method), True)
        self.assertEqual(type(rgb_depth.get_pillow_image()), Image.Image)


class DetectionHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detect_handler = DetectionHandler(PERCEPTION_MODELS_DIR)

    def test_handler(self):
        # load the image and pass it to detectron2 model
        img = cv2.imread(OFFICE_IMG_PATH, 1)
        rgb_depth_mock = MagicMock()
        rgb_depth_mock.rgb = img
        detections = self.detect_handler.handle(rgb_depth_mock)

        # check that most of the detected objects are detected
        self.assertGreaterEqual(len(detections), 15)  # 17 exactly

        # check the correct type of each detected object
        self.assertEqual(type(detections[0]), Detection)


class MemoryHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detect_handler = DetectionHandler(PERCEPTION_MODELS_DIR)
        self.agent = MagicMock()
        self.agent.memory = LocoAgentMemory()
        self.memory = MemoryHandler(self.agent)
        self.deduplicator = ObjectDeduplicationHandler()

    def test_handler_basic_insertion(self):
        # get fake detection
        r = get_fake_rgbd()
        props = ["dragon heartstring", "phoenix feather", "coral"]
        d = get_fake_detection("wand", props, [0, 0, 0])
        # get fake human pose
        h = get_fake_humanpose()
        # save to memory
        self.memory([d, h], [])

        # retrieve detected objects
        o = DetectedObjectNode.get_all(self.agent.memory)
        self.assertEqual(len(o), 1)
        self.assertEqual(o[0]["label"], "wand")
        self.assertEqual(o[0]["properties"], str(props))

        # retrieve human poses
        rh = HumanPoseNode.get_all(self.agent.memory)
        self.assertEqual(len(rh), 1)
        self.assertEqual(rh[0]["keypoints"], h.keypoints)

    def test_handler_dedupe(self):
        # load the image and pass it to detectron2 model
        img = cv2.imread(OFFICE_IMG_PATH, 1)
        rgbd = get_fake_rgbd(rgb=img)
        logging.getLogger().disabled = True
        detections = self.detect_handler.handle(rgbd)

        # check that most of the detected objects are detected
        self.assertGreaterEqual(len(detections), 15)  # 17 exactly
        # insert once to setup dedupe tests
        self.deduplicator(detections, [])
        self.memory(detections, [])
        objs_init = DetectedObjectNode.get_all(self.agent.memory)

        # Insert with dedupe
        previous_objects = self.memory.get_objects()
        if previous_objects is not None:
            new_objects, updated_objects = self.deduplicator(detections, previous_objects)
            self.memory(new_objects, updated_objects)
        
        # Assert that some objects get deduped
        objs_t1 = DetectedObjectNode.get_all(self.agent.memory)
        self.assertLessEqual(len(objs_t1), len(objs_init) + len(detections))

        logging.getLogger().disabled = False
        logging.info("Number of detections {}".format(len(detections)))


class TestFaceRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.f_rec = FaceRecognitionHandler(FACES_IDS_DIR)

    def test_creation(self):
        """class can be created and faces loaded successfully."""
        self.assertFalse(
            self.f_rec.faces_path is None,
            "Faces IDs path shouldn't be empty <every face will be encoded as 'unknown'>",
        )

    def test_encoded_faces(self):
        """check if there are faces IDs."""
        self.assertTrue(len(self.f_rec.encoded_faces) > 1, "There should be 8 encoded faces IDs")

    def test_many_faces(self):
        """class is able to detect: more than one face."""
        group_img = cv2.imread(GROUP_IMG_PATH, 1)
        rgb_depth_mock = MagicMock()
        rgb_depth_mock.rgb = group_img
        self.f_rec.detect_faces(rgb_depth_mock)
        self.assertTrue(len(self.f_rec.face_locations) >= 3, "detected less than 50% of the faces")

    def test_no_faces(self):
        """class is able to work: with zero number of faces."""
        office_img = cv2.imread(OFFICE_IMG_PATH, 1)
        rgb_depth_mock = MagicMock()
        rgb_depth_mock.rgb = office_img
        self.f_rec.detect_faces(rgb_depth_mock)
        self.assertTrue(len(self.f_rec.face_locations) == 0, "detected face in a no-face image!")


if __name__ == "__main__":
    unittest.main()
