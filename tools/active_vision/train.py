# Below is a copy of slurm_train.py from anurag/slurm_onebutton

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
from detectron2.utils.events import EventWriter, get_event_storage
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
from pycocotools.coco import COCO

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

## Detectron2 Setup

from detectron2.engine.hooks import HookBase, EvalHook
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging

class LossEvalHook(HookBase):
    def __init__(self, cfg, model, data_loader):
        self._model = model
        self._period = cfg.TEST.EVAL_PERIOD,
        self._data_loader = data_loader
        self.cfg = cfg
        print(f'self._period {self._period}')
        if type(self._period) == tuple:
            self._period = self._period[0]
            print(f"self_period was a tuple, now {self._period}")

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        print(f'mean_loss {mean_loss}, len(losses) {len(losses)}')
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        log_every_n_seconds(logging.INFO, f"Validation loss dict {metrics_dict}", n=5)
        # print(f"Validation loss dict {metrics_dict}")
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def _do_test_eval(self):
        with open(os.path.join(self.cfg.OUTPUT_DIR, "test_results.txt"), "a") as f:
            f.write(f'lr {self.cfg.SOLVER.BASE_LR} warmup {self.cfg.SOLVER.WARMUP_ITERS} iter {self.trainer.iter}\n')
            for yix in range(len(test_jsons)):
                output_folder = os.path.join(self.cfg.OUTPUT_DIR, "test_inference")
                evaluator = COCOEvaluator('test' + str(yix), ("bbox", "segm"), False, self.cfg.OUTPUT_DIR, use_fast_impl=True)
                data_loader = build_detection_test_loader(
                    DatasetCatalog.get('test' + str(yix)),
                    mapper=DatasetMapper(self.cfg, is_train=False)
                )
                results = inference_on_dataset(self._model, data_loader, evaluator)
                f.write(json.dumps(results) + '\n')
        
    def after_step(self):
        # print(f'self._period {self._period} next_iter {self.trainer.iter}....')
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            losses = self._do_loss_eval()
            l = np.mean(losses)
            # write validation loss, AP
            print(f'val los {self.trainer.iter} losses.mean {l}')
            with open(os.path.join(self.cfg.OUTPUT_DIR, "validation_results.txt"), "a") as f:
                f.write(f'validation loss {l}\n')
            with open(os.path.join(self.cfg.OUTPUT_DIR, "validation_results2.txt"), "a") as f:
                f.write(f'lr {self.cfg.SOLVER.BASE_LR} warmup {self.cfg.SOLVER.WARMUP_ITERS} iter {self.trainer.iter}\n')
                f.write(f'validation loss {l}\n')
                output_folder = os.path.join(self.cfg.OUTPUT_DIR, "val_inference")
                evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_folder)
                results = inference_on_dataset(self._model, self._data_loader, evaluator)
                f.write(json.dumps(results) + '\n')

            # write test AP 
            # self._do_test_eval()

        self.trainer.storage.put_scalars(timetest=12)
        

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN, max_size=1333, sample_style='choice'),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute", (640, 640)),
            T.RandomBrightness(0.9, 1.1)
        ])
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        results = super().test(cfg, model, evaluators)
        print(results)
        # save to file here to compare with saving inside the hook
        with open(os.path.join(cfg.OUTPUT_DIR, "validation_results.txt"), "a") as f:
            f.write(json.dumps(results) + '\n')
        return results

    def build_hooks(self):
        hooks = super().build_hooks()
        print(f'hooks {len(hooks), hooks}')
        hooks.insert(-2,LossEvalHook(
            self.cfg,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, is_train=True)
            ),
        ))
        print(f'hooks {len(hooks), hooks}')
        return hooks


class COCOTrain:
    def __init__(self, lr, w, maxiters, seed, data):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(coco_yaml))
        self.cfg.SOLVER.BASE_LR = lr  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = maxiters
        self.cfg.SOLVER.WARMUP_ITERS = w
        self.seed = seed
        self.data = data
    
    def register_json(self, name, coco_json, img_dir):
        register_coco_instances(name, {}, coco_json, img_dir)
        # roundabout way to set metadata https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#metadata-for-datasets
        coco = COCO(coco_json)
        cats = coco.loadCats(coco.getCatIds())
        MetadataCatalog.get(name).thing_classes = [cat['name'] for cat in cats]
        
    def reset(self):
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        self.train_data = self.data['dataset_name'] +  "_train"
        self.register_json(self.train_data, self.data['train']['json'], self.data['train']['img_dir'])
        if 'val' in self.data.keys():
            self.val_data = self.data['dataset_name'] + "_val"
            self.register_json(self.val_data, self.data['val']['json'], self.data['val']['img_dir'])
        
        self.test_data = None
        if 'test' in self.data.keys():
            self.test_data = self.data['dataset_name'] + "_test"
            self.register_json(self.val_data, self.data['test']['json'], self.data['test']['img_dir'])
    
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
        cfg.DATASETS.TEST = (self.val_data, self.train_data if not self.test_data else self.test_data)
        
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_yaml)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 16
        cfg.TEST.EVAL_PERIOD = 50
        cfg.SOLVER.GAMMA=0.75
        cfg.SOLVER.STEPS=tuple([100*(i+1) for i in range(100) if 100*(i+1) < cfg.SOLVER.MAX_ITER])
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
        print(f'classes {MetadataCatalog.get(self.train_data)}')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(self.train_data).get("thing_classes"))  
        cfg.OUTPUT_DIR = os.path.join(self.data['out_dir'], 'training', str(self.seed), str(cfg.SOLVER.MAX_ITER), str(cfg.SOLVER.BASE_LR), str(cfg.SOLVER.WARMUP_ITERS))
        print(f"recreating {cfg.OUTPUT_DIR}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = MyTrainer(cfg) #DefaultTrainer(cfg)  #Trainer(cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        
    def run_train(self):
        self.reset()
        # self.vis()
        self.train()


maxiters = [1000]
lrs = [0.001, 0.002, 0.004]
warmups = [100]

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

def run_training(training_data, n=1):
    for lr in lrs:
        for warmup in warmups:
            for maxiter in maxiters:
                for i in range(n):
                    c = COCOTrain(lr, warmup, maxiter, i, training_data)
                    c.run_train()