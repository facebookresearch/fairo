"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import sys
import cv2
import numpy as np
import pickle
import tempfile
import logging
from imantics import Mask
import torch
import json

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import droidlet.memory.robot.loco_memory as loco_memory

from .core import AbstractHandler, WorldObject
from droidlet.shared_data_structs import RGBDepth
from ..detectron.detector.utils import get_predictor
from ..detectron.detector.visualizer import LocobotVisualizer
from droidlet.perception.robot.perception_helpers import get_color_tag


lvis_yaml = "configs/mask_rcnn_R_101_FPN_1x.yaml"
detector_weights = "model_999.pth"
default_json_dir = os.path.abspath(os.path.dirname(__file__)) + "/../../../../annotation_data/model"
props_filename = "props.json"
things_filename = "things.json"

file_root = os.path.dirname(os.path.realpath(__file__))


class ObjectDetection(AbstractHandler):
    """Class for object detector.

    We use a modified Mask R-CNN with an additional head that predicts object properties.

    Args:
        model_data_dir (string): path to the model directory
    """

    def __init__(self, model_data_dir):
        self.detector = DetectorBase(model_data_dir)

    def __call__(self, rgb_depth):
        """the inference logic for the handler lives here.

        Args:
            rgb_depth (RGBDepth): the current input frame to run inference on.

        Returns:
            detections (list[Detections]): list of detections found
        """
        logging.info("In DetectionHandler ... ")
        rgb = rgb_depth.rgb
        p_list, predictions = self.detector(rgb)
        detections = []
        for x in p_list:
            logging.info("Detected {} objects".format(len(p_list)))
            # create a detection object for each instance
            detections.append(
                Detection(
                    rgb_depth,
                    x["class_label"],
                    x["prop_label"],
                    x["mask"],
                    x["bbox"],
                    center=x["center"],
                )
            )
        if os.getenv("DEBUG_DRAW") == "True":
            self._debug_draw(rgb_depth, predictions)
        return detections

    def _debug_draw(self, rgb_depth, predictions):
        self.detector.draw(rgb_depth.rgb, predictions)


class DetectorBase:
    """Class that encapsulates low_level logic for the detector, like loading the model and parsing inference outputs."""

    def __init__(self, model_data_dir):
        files = os.listdir(model_data_dir)
        props_file = os.path.join(model_data_dir, props_filename) \
            if props_filename in files \
            else os.path.join(default_json_dir, props_filename)
        things_file = os.path.join(model_data_dir, things_filename) \
            if things_filename in files \
            else os.path.join(default_json_dir, things_filename)

        print("cheesecake", os.path.abspath(os.path.dirname(__file__)), os.listdir(default_json_dir))

        with open(props_file, "r") as h:
            self.properties = json.load(h)["items"]
            logging.info("{} properties".format(len(self.properties)))
        with open(things_file, "r") as h:
            self.things = json.load(h)["items"]
            logging.info("{} things".format(len(self.things)))

        weights = os.path.join(model_data_dir, detector_weights)
        self.dataset_name = "dummy_dataset"
        self.predictor = get_predictor(
            lvis_yaml, weights, self.dataset_name, self.properties, self.things
        )

    def __call__(self, img):
        predictions = self.predictor(img)
        oi = predictions["instances"].to("cpu").get_fields()
        centers = oi[
            "pred_boxes"
        ].get_centers()  # N*2 https://detectron2.readthedocs.io/_modules/detectron2/structures/boxes.html
        p_list = []
        num_instances = len(oi["pred_classes"])
        logging.info("{} instances detected.".format(num_instances))
        for x in range(num_instances):
            p = {}
            # class label
            pred_class = oi["pred_classes"][x]
            pred_label = self.things[pred_class]
            p["class_label"] = pred_label

            # properties
            pred_prop = oi["pred_props"][x]
            prop_label = []
            for k in pred_prop:
                prop_label.append(self.properties[k])
            p["prop_label"] = prop_label

            # mask
            p["mask"] = oi["pred_masks"][x]
            p["bbox"] = oi["pred_boxes"][x]
            p["center"] = (int(centers[x][0].item()), int(centers[x][1].item()))
            p_list.append(p)

        return p_list, predictions

    def draw(self, im, predictions, save_to_disk=False):
        v = LocobotVisualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(self.dataset_name),
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imshow("Insight", v.get_image()[:, :, ::-1])
        cv2.waitKey(3)

        if save_to_disk:
            tmp = os.path.join(file_root, "tmp")
            if not os.path.isdir(tmp):
                os.mkdir(tmp)
            tf = tempfile.NamedTemporaryFile(
                prefix="detector_", suffix=".jpg", dir=tmp, delete=False
            )
            logging.info("Saving to {}".format(tf.name))
            cv2.imwrite(tf.name, v.get_image()[:, :, ::-1])


class Detection(WorldObject):
    """Instantiation of the WorldObject that is used by the detector."""

    def __init__(
        self,
        rgb_depth: RGBDepth,
        class_label,
        properties,
        mask,
        bbox,
        face_tag=None,
        center=None,
        xyz=None,
    ):
        WorldObject.__init__(
            self, label=class_label, center=center, rgb_depth=rgb_depth, mask=mask, xyz=xyz
        )
        self.bbox = bbox
        self.tracked_features = []
        self.properties = properties
        self.color = get_color_tag(rgb_depth.get_pillow_image(), self.center)
        self.facial_rec_tag = face_tag
        self.feature_repr = None

    def save_to_memory(self, memory, update=False):
        if update:
            loco_memory.DetectedObjectNode.update(memory, self)
        else:
            loco_memory.DetectedObjectNode.create(memory, self)

    def _maybe_bbox(self, bbox, mask):
        if hasattr(bbox, "tensor"):
            bbox = bbox.tensor.tolist()[0]
        if bbox is None:
            nz = mask.nonzero()
            y, x = nz[0], nz[1]
            bbox = [int(x[0]), int(y[0]), int(x[-1]), int(y[-1])]
        return bbox

    def to_struct(self):
        bbox = self._maybe_bbox(self.bbox, self.mask)
        mask_arr = []
        if self.mask is not None and isinstance(self.mask, np.ndarray):
            mask_arr =  self.mask
        if self.mask is not None and isinstance(self.mask, torch.Tensor): 
            mask_arr = self.mask.cpu().detach().numpy()
        mask_points_nd = Mask(mask_arr).polygons().points
        mask_points = list(map(lambda x: x.tolist(), mask_points_nd))
        return {
            "id": self.eid,
            "xyz": list(self.xyz),
            "bbox": bbox,
            "label": self.label,
            "properties": "\n ".join(self.properties if self.properties is not None else ""),
            "mask": mask_points,
        }

    def get_masked_img(self):
        rgb = self.rgb_depth.rgb
        h, w, _ = rgb.shape
        bbox = self._maybe_bbox(self.bbox, self.mask)
        x1, y1, x2, y2 = [int(x) for x in bbox]
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1, 4)
        im = cv2.bitwise_and(rgb, rgb, mask=mask)
        logging.debug("Calculating feature repr for {}".format(self.label))
        return im
