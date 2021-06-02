"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import cv2
import os
import logging
import droidlet.memory.robot.loco_memory as loco_memory
import numpy as np
from .core import AbstractHandler, WorldObject
from droidlet.shared_data_structs import RGBDepth
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from collections import namedtuple

file_root = os.path.dirname(os.path.realpath(__file__))


class HumanPose(AbstractHandler):
    """Class for Human Pose estimation.

    We use a keypoint estimator.

    Args:
        model_data_dir (string): path to the model directory
    """

    def __init__(self, model_data_dir):
        self.detector = HumanKeypoints(model_data_dir)

    def forward(self, rgb_depth):
        logging.info("In HumanPoseHandler ... ")
        rgb = rgb_depth.rgb
        keypoints = self.detector(rgb)
        humans = []
        for i in range(len(keypoints)):
            humans.append(Human(rgb_depth, keypoints[i]))

        # self.draw(rgb_depth, keypoints)
        return humans

    def _debug_draw(self, rgb_depth, keypoints):
        self.detector.draw(rgb_depth.rgb, keypoints)


keypoints_yaml = "detectron/configs/Keypoints/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
keypoints_weights = "keypoints_model_final_a6e10b.pkl"

COCO_PERSON_KEYPOINT_ORDERING = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class HumanKeypoints:
    draw_lines = [
        ("left_ear", "left_eye"),
        ("right_ear", "right_eye"),
        ("left_eye", "right_eye"),
        ("left_eye", "nose"),
        ("right_eye", "nose"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_shoulder", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_shoulder", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_hip", "left_hip"),
        ("right_shoulder", "left_shoulder"),
    ]

    def __init__(self, model_data_dir):
        cfg = get_cfg()
        yaml_path = os.path.abspath(os.path.join(file_root, "..", keypoints_yaml))
        cfg.merge_from_file(yaml_path)
        weights = os.path.join(model_data_dir, keypoints_weights)
        cfg.MODEL.WEIGHTS = weights
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        predictions = self.predictor(img)
        predictions = predictions["instances"]

        points_indices = predictions.pred_keypoints
        scores = predictions.scores
        points = []
        for i in range(len(points_indices)):
            if scores[i] >= 0.8:
                p_i = {}
                for point, name in zip(points_indices[i], COCO_PERSON_KEYPOINT_ORDERING):
                    p_i[name] = point.int().tolist()
                points.append(HumanKeypointsOrdering(**p_i))

        logging.info("{} humans detected.".format(len(points)))
        return points

    def draw(self, bgr, keypoints):
        rgb = bgr[:, :, ::-1]
        for i in range(len(keypoints)):
            points_i = keypoints[i]
            for d in self.draw_lines:
                p1 = getattr(points_i, d[0])
                p2 = getattr(points_i, d[1])
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                rgb = cv2.line(rgb, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=3)
        cv2.imshow("Humans", rgb)
        cv2.waitKey(3)


HumanKeypointsOrdering = namedtuple(
    "HumanKeypoints",
    [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
)


class Human(WorldObject):
    """Instantiation of the WorldObject that is used by the human pose estimator."""

    def __init__(self, rgb_depth: RGBDepth, keypoints: HumanKeypointsOrdering):
        WorldObject.__init__(self, label="human_pose", center=keypoints.nose[:2], rgb_depth=rgb_depth)
        self.keypoints = keypoints

    def save_to_memory(self, agent):
        loco_memory.HumanPoseNode.create(agent.memory, self)

    def to_struct(self):
        return {"xyz": list(self.xyz), "keypoints": self.keypoints._asdict()}

    # FIXME - currently just returns a random rgb image
    def get_masked_img(self):
        h, w, _ = self.rgb_depth.rgb.shape
        return np.float32(np.random.rand(h, w, 3) * 255)
