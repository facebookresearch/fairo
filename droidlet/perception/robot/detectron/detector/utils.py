"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import os

dirname = os.path.dirname(os.path.realpath(__file__))


def add_property_config(cfg):
    """Add config for property head."""
    _C = cfg
    _C.MODEL.DENSEPOSE_ON = True
    _C.MODEL.ROI_PROPERTY_HEAD = CN()
    _C.MODEL.ROI_PROPERTY_HEAD.NAME = ""
    _C.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES = 8


def get_dicts():
    pass


def get_predictor(config, model_weights, dataset, properties, things):
    cfg = get_cfg()
    add_property_config(cfg)
    cfg.merge_from_file(os.path.join(dirname, config))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(things)
    DatasetCatalog.clear()
    DatasetCatalog.register(dataset, get_dicts)
    MetadataCatalog.get(dataset).set(thing_classes=things)
    MetadataCatalog.get(dataset).set(property_classes=properties)
    cfg.DATASETS.TEST = (dataset,)
    return DefaultPredictor(cfg)
