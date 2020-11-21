#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Locobot Detector Training Script.

This script is a simplified version of the training script in
detectron2/tools.
"""

import os
import pickle
import logging
import glob
import random
import detectron2.utils.comm as comm
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import json
import torch
from detectron2.structures.boxes import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.modeling import ROI_HEADS_REGISTRY
from utils import add_property_config
from roi_heads import LocobotROIHeads
from dataset_mapper import LocobotDatasetMapper

# ROI_HEADS_REGISTRY.register(LocobotROIHeads)


class Trainer(DefaultTrainer):
    # FIXME add custom evaluator for property and detector metrics
    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = LocobotDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_property_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "LVIS-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset (default: 512)
    )

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_dataset_dicts(set_name, data_dir, json_file):
    """This function tells detectron2 how to fetch your custom dataset.

    https://detectron2.readthedocs.io/tutorials/datasets.html
    """
    botfiles = glob.glob(os.path.join(data_dir, "botcapture*.jpg"))
    random.shuffle(botfiles)

    def slc(files, set_name, pct):
        l = int(pct * len(files))
        return files[-l:] if set_name == "Val" else files[:-l]

    train_files = slc(botfiles, "Train", 0.1)
    val_files = slc(botfiles, "Val", 0.1)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    files = train_files if set_name == "train" else val_files
    print("number of files {}".format(len(files)))
    for idx, filepath in enumerate(files):
        record = {}
        height, width = cv2.imread(filepath).shape[:2]
        record["file_name"] = filepath
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        filename = filepath.split("/")[-1]
        objs = []
        keep_rec = False
        for _, anno in enumerate(imgs_anns):
            if anno["image_id"] == filename or str(anno["image_id"]) + ".jpg" == filename:
                keep_rec = True
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": anno["segmentation"],
                    "category_id": anno["category_id"],
                    "iscrowd": 0,
                    "properties": torch.tensor([x for x in anno["properties"]]).long(),
                }
                objs.append(obj)

        record["annotations"] = objs
        if keep_rec:
            dataset_dicts.append(record)
    return dataset_dicts


# The expected dataset directory structure for properties + object detector is as follows:
# \data_dir
#   \prop.pickle
#   \things.pickle
#   \annotations.json
#   \Images


def setup_dataset(args):
    dataset = args.dataset_name
    data_dir = args.data_dir
    anno_json = args.annotations_json
    with open(os.path.join(data_dir, "prop.pickle"), "rb") as h:
        properties = pickle.load(h)
        logging.info("{} properties".format(len(properties)))

    with open(os.path.join(data_dir, "things.pickle"), "rb") as h:
        things = pickle.load(h)
        logging.info("{} things".format(len(things)))

    for d in ["_train", "_val"]:
        DatasetCatalog.register(dataset + d, lambda d=d: get_dataset_dicts(d, data_dir, anno_json))
        MetadataCatalog.get(dataset + d).set(thing_classes=things)
        MetadataCatalog.get(dataset + d).set(property_classes=properties)


def main(args):
    setup_dataset(args)
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--data-dir", help="root folder of the dataset directory")
    args.add_argument(
        "--dataset-name", help="dataset named used for registering to the DatasetCatalog"
    )
    args.add_argument("--annotations-json", help="path to annotations json file")
    args = args.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
