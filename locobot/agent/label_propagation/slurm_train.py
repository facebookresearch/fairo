import torchvision

# import some common libraries
import numpy as np
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import random
import os
import numpy as np
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
import shutil
from setuptools.namespaces import flatten

import random
import torch 
import base64
import io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
from PIL import Image


coco_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
lvis_yaml = "LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
lvis_yaml2 = "LVIS-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"
pano_yaml = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

jsons_root = '/checkpoint/apratik/finals/jsons/active_vision/'
img_dir_test = '/checkpoint/apratik/ActiveVision/active_vision/replica_random_exploration_data/frl_apartment_1/rgb'
test_jsons = ['frlapt1_20n0.json', 'frlapt1_20n1.json', 'frlapt1_20n2.json']
test_jsons = [os.path.join(jsons_root, x) for x in test_jsons]

## Detectron2 Setup

# from copy_paste import CopyPaste
# import albumentations as A

class Trainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN, max_size=1333, sample_style='choice'),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute", (640, 640)),
            T.RandomBrightness(0.9, 1.1)
        ])
        return build_detection_train_loader(cfg, mapper=mapper)

class COCOTrain:
    def __init__(self, lr, w, maxiters, seed):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(coco_yaml))
        self.cfg.SOLVER.BASE_LR = lr  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = maxiters
        self.cfg.SOLVER.WARMUP_ITERS = w
        self.seed = seed
        
    def reset(self, train_json, img_dir_train, dataset_name):
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        self.train_data = dataset_name +  "_train"
        self.dataset_name = dataset_name
        self.train_json = train_json
        register_coco_instances(self.train_data, {}, train_json, img_dir_train)
        self.results = {
            "bbox": {
                "AP50": []
            },
            "segm": {
                "AP50": []
            }
        }
    
    def vis(self):
        dataset_dicts = DatasetCatalog.get(self.train_data)
        for d in random.sample(dataset_dicts, 2):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(self.train_data), scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            img = vis.get_image()
            plt.figure(figsize=(12,8))
            plt.imshow(img)
            plt.show()
            
    def train(self):
        cfg = self.cfg
        print(f'SOLVER PARAMS {cfg.SOLVER.MAX_ITER, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.BASE_LR}')
        cfg.DATASETS.TRAIN = (self.train_data,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_yaml)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
        MetadataCatalog.get(self.train_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        print(f'classes {MetadataCatalog.get(self.train_data)}')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(self.train_data).get("thing_classes"))  
        cfg.OUTPUT_DIR = os.path.join('output_aug', str(cfg.SOLVER.MAX_ITER), self.dataset_name + str(self.seed))
        print(f"recreating {cfg.OUTPUT_DIR}")
        # if os.path.isdir(cfg.OUTPUT_DIR):
        #     shutil.rmtree(cfg.OUTPUT_DIR)
        print(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = Trainer(cfg) #DefaultTrainer(cfg) 
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()

    def run_eval(self, dataset_name, test_json, img_dir_test):
        self.val_data = dataset_name + "_val" + str(self.seed)
        self.test_json = test_json
        self.cfg.DATASETS.TEST = (self.val_data,)
        register_coco_instances(self.val_data, {}, test_json, img_dir_test)
        MetadataCatalog.get(self.val_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        print(f'classes {MetadataCatalog.get(self.val_data)}')
        self.evaluator = COCOEvaluator(self.val_data, ("bbox", "segm"), False, output_dir=self.cfg.OUTPUT_DIR)
        self.val_loader = build_detection_test_loader(self.cfg, self.val_data)
        results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
        self.results['bbox']['AP50'].append(results['bbox']['AP50'])
        self.results['segm']['AP50'].append(results['segm']['AP50'])
        return results
        
    def run_train(self, train_json, img_dir_train, dataset_name):
        self.reset(train_json, img_dir_train, dataset_name)
        # self.vis()
        self.train()


maxiters = 500
lr = [0.001, 0.002, 0.005]
warmup = [100, 200]

def write_summary_to_file(filename, results, header_str):
    if isinstance(results['bbox']['AP50'][0], list):
        results['bbox']['AP50'] = list(flatten(results['bbox']['AP50']))
        results['segm']['AP50'] = list(flatten(results['segm']['AP50']))
    with open(filename, "a") as f:
        f.write(header_str)
        f.write(f"\nbbox AP50 {sum(results['bbox']['AP50'])/len(results['bbox']['AP50'])}")
        f.write(f"\nsegm AP50 {sum(results['segm']['AP50'])/len(results['segm']['AP50'])}")
        f.write(f'\nall results {results}')

from pathlib import Path
import string

def run_training(out_dir, img_dir_train, n=10):
    train_json = os.path.join(out_dir, 'coco_train.json')
    results = {
        "bbox": {
            "AP50": []
        },
        "segm": {
            "AP50": []
        }
    }
    for i in range(n):
        c = COCOTrain(lr[0], warmup[0], maxiters, i)
        dataset_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
        print(f'dataset_name {dataset_name}')
        c.run_train(train_json, img_dir_train, dataset_name)
        for yix in range(len(test_jsons)):
            r = c.run_eval(dataset_name + str(yix), test_jsons[yix], img_dir_test)
            with open(os.path.join(out_dir, "all_results.txt"), "a") as f:
                f.write(json.dumps(r))
        print(f'all results {c.results}')
        results['bbox']['AP50'].append(c.results['bbox']['AP50'])
        results['segm']['AP50'].append(c.results['segm']['AP50'])
        write_summary_to_file(os.path.join(out_dir, str(n) + '_granular.txt'), c.results, f'\ntrain_json {train_json}')

    write_summary_to_file(os.path.join(out_dir, str(n) + '_results_averaged.txt'), results, f'\ntrain_json {train_json}, average over {n} runs')
   