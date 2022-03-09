"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import unittest
import logging
from timeit import Timer
from unittest.mock import MagicMock
from droidlet.perception.robot import (
    ObjectDetection,
    FaceRecognition,
    ObjectDeduplicator,
    Perception,
    Detection,
    LabelPropagate,
)
from droidlet.interpreter.robot import dance
from droidlet.memory.robot.loco_memory import LocoAgentMemory
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode, HumanPoseNode
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
from droidlet.shared_data_struct.robot_shared_utils import RobotPerceptionData
import cv2
import torch
from PIL import Image
from droidlet.perception.robot.tests.utils import (
    get_fake_rgbd,
    get_fake_detection,
    get_fake_humanpose,
)
from droidlet.perception.robot.active_vision.candidate_selection import SampleGoodCandidates
import json
import numpy as np
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IP = "127.0.0.1"
if os.getenv("LOCOBOT_IP"):
    IP = os.getenv("LOCOBOT_IP")

PERCEPTION_MODELS_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "../../../../droidlet/artifacts/models/perception/locobot",
)

OFFICE_IMG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "../../../../droidlet/artifacts/datasets/robot/perception_test_assets",
    "office_chair.jpg",
)
GROUP_IMG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "../../../../droidlet/artifacts/datasets/robot/perception_test_assets",
    "obama_trump.jpg",
)
FACES_IDS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../droidlet/artifacts/datasets/robot/perception_test_assets/faces",
)

CANDIDATE_SELECTION_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../droidlet/artifacts/datasets/robot/perception_test_assets/candidate_selection_data",
)

LABEL_PROP_TEST_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../droidlet/artifacts/datasets/robot/perception_test_assets/label_prop_test_assets",
)

logging.getLogger().setLevel(logging.INFO)


class LabelPropTest(unittest.TestCase):
    def setUp(self):
        self.lp = LabelPropagate()
        self.test_assets = LABEL_PROP_TEST_ASSETS_DIR

    def read_test_asset_idx(self, root, img_indx):
        src_img = cv2.imread(os.path.join(root, "rgb_{:05d}.jpg".format(img_indx)))
        src_depth = np.load(os.path.join(root, "depth_{:05d}.npy".format(img_indx)))
        src_label = np.load(os.path.join(root, "seg_{:05d}.npy".format(img_indx)))
        with open(os.path.join(root, "data.json"), "r") as f:
            base_pose_data = json.load(f)
        src_pose = base_pose_data["{}".format(img_indx)]

        # Visualize label
        return src_img, src_label, src_depth, src_pose, None

    def calculate_accuracy(self, act, pred):
        h, w = act.shape
        assert act.shape == pred.shape

        correct = np.sum(act[pred != 0] == pred[pred != 0])
        total = np.sum(pred != 0)

        return correct / total

    def _run_test(self, data_dir):
        """
        Checks that each label prop call runs in < 0.1 seconds with > 90% accuracy
        """
        lp = LabelPropagate()
        for x in os.listdir(data_dir):
            dd = os.path.join(data_dir, x)

            # Each test asset folder has one source id, and one id to label propagate to
            ids = []
            with open(os.path.join(dd, "gtids.txt"), "r") as f:
                ids = f.readlines()
                ids = [int(x.strip()) for x in ids]

            src_img, src_label, src_depth, src_pose, cam_transform = self.read_test_asset_idx(
                dd, ids[0]
            )
            cur_img, cur_label, cur_depth, cur_pose, cam_transform = self.read_test_asset_idx(
                dd, ids[1]
            )

            start = time.time()
            prop_label = self.lp(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)
            time_taken = time.time() - start
            logging.info(f"time taken {time_taken}")
            acc = self.calculate_accuracy(cur_label, prop_label)
            assert acc * 100 > 90, f"accuracy {acc} < 90"

    def test_label_prop_nonoise(self):
        data_dir = os.path.join(self.test_assets, "no_noise")
        self._run_test(data_dir)

    def test_label_prop_noise(self):
        data_dir = os.path.join(self.test_assets, "noise")
        self._run_test(data_dir)


class PerceiveTimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mover = LoCoBotMover(ip=IP, backend="habitat")
        self.perception = Perception(PERCEPTION_MODELS_DIR)

    def test_time(self):
        rgb_depth = self.mover.get_rgb_depth()
        xyz = (0, 0, 0)
        previous_objects = []
        t = Timer(lambda: self.perception.perceive(rgb_depth, xyz, previous_objects, force=True))
        logging.info("Perception runtime {} s".format(t.timeit(number=1)))


class DetectionHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detect_handler = ObjectDetection(PERCEPTION_MODELS_DIR)

    def test_handler(self):
        # load the image and pass it to detectron2 model
        img = cv2.imread(OFFICE_IMG_PATH, 1)
        rgb_depth_mock = MagicMock()
        rgb_depth_mock.rgb = img
        detections = self.detect_handler.__call__(rgb_depth_mock)

        # check that most of the detected objects are detected
        self.assertGreaterEqual(len(detections), 5)  # 9 exactly

        # check the correct type of each detected object
        self.assertEqual(type(detections[0]), Detection)

    def test_detection_to_struct(self):
        # use case: face detections have no mask but are still Detections
        d = get_fake_detection("", "", "")
        d.mask = None

        # check whether to_struct fails
        try:
            d.to_struct()
        except:
            self.fail("detection's to_struct() fails when no mask is detected")


class MemoryStoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detect_handler = ObjectDetection(PERCEPTION_MODELS_DIR)
        self.agent = MagicMock()
        self.agent.memory = LocoAgentMemory()
        dance.add_default_dances(self.agent.memory)
        self.deduplicator = ObjectDeduplicator()

    def test_basic_insertion(self):
        # get fake detection
        r = get_fake_rgbd()
        props = ["dragon heartstring", "phoenix feather", "coral"]
        d = get_fake_detection("wand", props, [0, 0, 0])
        # get fake human pose
        h = get_fake_humanpose()
        # save to memory
        perception_output = RobotPerceptionData(new_objects=[d], humans=[h])
        self.agent.memory.update(perception_output)

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
        detections = self.detect_handler.__call__(rgbd)

        # check that most of the detected objects are detected
        self.assertGreaterEqual(len(detections), 5)  # 9 exactly
        # insert once to setup dedupe tests
        self.deduplicator(detections, [])
        self.agent.memory.update(RobotPerceptionData(new_objects=detections))

        objs_init = DetectedObjectNode.get_all(self.agent.memory)

        # Insert with dedupe
        previous_objects = DetectedObjectNode.get_all(self.agent.memory)
        if previous_objects is not None:
            new_objects, updated_objects = self.deduplicator(detections, previous_objects)
            detection_output = RobotPerceptionData(
                new_objects=new_objects, updated_objects=updated_objects
            )
            self.agent.memory.update(detection_output)

        # Assert that some objects get deduped
        objs_t1 = DetectedObjectNode.get_all(self.agent.memory)
        self.assertLessEqual(len(objs_t1), len(objs_init) + len(detections))

        logging.getLogger().disabled = False
        logging.info("Number of detections {}".format(len(detections)))


class TestFaceRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.f_rec = FaceRecognition(FACES_IDS_DIR)

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


class TestCandidateSelection(unittest.TestCase):
    def load_semantic_json(self, scene):
        habitat_semantic_json = os.path.join(
            CANDIDATE_SELECTION_DIR, scene, "habitat", "info_semantic.json"
        )
        with open(habitat_semantic_json, "r") as f:
            hsd = json.load(f)
        if hsd is None:
            print("Semantic json not found!")
        return hsd

    def _run_test(self, traj_path, good_candidates, bad_candidates, is_annot_validfn):
        s = SampleGoodCandidates(traj_path, is_annot_validfn)

        # Get good candidates
        good = s.get_n_candidates(5, good=True)
        assert np.array_equal(np.asarray(good.sort()), np.asarray(good_candidates.sort()))

        # Get bad candidates
        bad = s.get_n_candidates(5, good=False)
        assert np.array_equal(np.asarray(bad.sort()), np.asarray(bad_candidates.sort()))

    def test_samplecandidates_for_class(self):
        """
        Tests candidate selection for the class setting (we care about instances belonging to labels)
        """
        traj_path = f"{CANDIDATE_SELECTION_DIR}/class"
        good_candidates = [474, 953, 255, 10, 9]
        bad_candidates = [393, 38, 76, 739, 174]

        def is_annot_validfn(annot):
            hsd = self.load_semantic_json("apartment_0")
            labels = ["chair", "cushion", "door", "indoor-plant", "sofa", "table"]
            label_id_dict = {}
            for obj_cls in hsd["classes"]:
                if obj_cls["name"] in labels:
                    label_id_dict[obj_cls["id"]] = obj_cls["name"]
            if (
                hsd["id_to_label"][annot] < 1
                or hsd["id_to_label"][annot] not in label_id_dict.keys()
            ):
                return False
            return True

        self._run_test(traj_path, good_candidates, bad_candidates, is_annot_validfn)

    def test_samplecandidates_for_instance(self):
        """
        Tests candidate selection for the instance setting (we care about instance ids in instance_ids)
        """
        traj_path = f"{CANDIDATE_SELECTION_DIR}/instance"
        instance_ids = [243, 404, 196, 133, 166, 170, 172]
        good_candidates = [117, 122, 103, 111, 100]
        bad_candidates = [41, 37, 60, 57, 35]

        def is_annot_validfn(annot):
            if annot not in instance_ids:
                return False
            return True

        self._run_test(traj_path, good_candidates, bad_candidates, is_annot_validfn)


if __name__ == "__main__":
    unittest.main()
