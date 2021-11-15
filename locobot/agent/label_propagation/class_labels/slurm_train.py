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

val_json0 = '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_train.json'
# val_json0 = '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_val20.json'
img_dir_val0 = '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/rgb'
# val_json = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/seg/coco_gt5_val.json'
# img_dir_val0 = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/rgb'
# val_json0 = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/seg/coco_gt5_4val.json'

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
        do_test_eval = False
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            losses = self._do_loss_eval()
            l = np.mean(losses)
            
            # get min val loss
            def get_min_val_loss(fname):
                min_val = 1000000
                with open(fname) as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), 3):
                        # best val
                        val_loss = float(lines[i+1].split()[-1])
                        if val_loss < min_val:
                            min_val = val_loss
                print(f'min_val {min_val} for {fname}')
                return min_val
            
            # if os.path.isfile(os.path.join(self.cfg.OUTPUT_DIR, "validation_results.txt")):
            #     if get_min_val_loss(os.path.join(self.cfg.OUTPUT_DIR, "validation_results.txt")) >= l:
            #         do_test_eval = True
            # else:
            #     do_test_eval = True

            # write validation loss, AP
            print(f'val los {self.trainer.iter} losses.mean {l}')
            print(f"writing to {os.path.join(self.cfg.OUTPUT_DIR, 'validation_results.txt')}")
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
            # if do_test_eval:
            #     self._do_test_eval()

        self.trainer.storage.put_scalars(timetest=12)
        
        
# class ActiveWriter(EventWriter):

#     def write(self):
#         storage = get_event_storage()
#         print(f'writing from ActiveWriter {storage.iter}')
#         print(storage)


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

    # def build_writers(self):
    #     writers = super().build_writers()
    #     print(f'current writers {writers}')
    #     writers.append(ActiveWriter())
    #     return writers

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
    def __init__(self, lr, w, maxiters, seed, name):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(coco_yaml))
        self.cfg.SOLVER.BASE_LR = lr  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = maxiters
        self.cfg.SOLVER.WARMUP_ITERS = w
        self.seed = seed
        self.name = name
        
    def reset(self, train_json, img_dir_train, dataset_name):
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        self.train_data = dataset_name +  "_train"
        self.dataset_name = dataset_name
        self.train_json = train_json
        register_coco_instances(self.train_data, {}, train_json, img_dir_train)

        # Register test json
        for yix in range(len(test_jsons)):
            register_coco_instances('test' + str(yix), {}, test_jsons[yix], img_dir_test)
            MetadataCatalog.get('test' + str(yix)).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
            print(MetadataCatalog.get('test' + str(yix)).thing_classes)
    
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
    
    def train(self, val_json, img_dir_val):
        cfg = self.cfg
        print(f'SOLVER PARAMS {cfg.SOLVER.MAX_ITER, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.BASE_LR}')
        cfg.DATASETS.TRAIN = (self.train_data,)
        
        # register validation set
        self.val_data = self.dataset_name + "_val" + str(self.seed)
        self.val_json = val_json
        register_coco_instances(self.val_data, {}, val_json, img_dir_val)
        MetadataCatalog.get(self.val_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        
        # register test set
        test_data = []
        for x in range(len(test_jsons)):
            tjson = test_jsons[x]
            tdata = self.dataset_name + "_test_" + str(self.seed) + '_' + str(x)
            test_data.append(tdata)
            register_coco_instances(tdata, {}, tjson, img_dir_test)
            MetadataCatalog.get(tdata).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']

        cfg.DATASETS.TEST = (self.val_data,test_data[0])

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_yaml)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 16
        cfg.TEST.EVAL_PERIOD = 50
        cfg.SOLVER.GAMMA=0.75
        cfg.SOLVER.STEPS=tuple([100*(i+1) for i in range(100) if 100*(i+1) < cfg.SOLVER.MAX_ITER])
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
        MetadataCatalog.get(self.train_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        print(f'classes {MetadataCatalog.get(self.train_data)}')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(self.train_data).get("thing_classes"))  
        cfg.OUTPUT_DIR = os.path.join('output_droid', self.name, str(self.seed), str(cfg.SOLVER.MAX_ITER), str(cfg.SOLVER.BASE_LR), str(cfg.SOLVER.WARMUP_ITERS))
        print(f"recreating {cfg.OUTPUT_DIR}")
        # if os.path.isdir(cfg.OUTPUT_DIR):
        #     shutil.rmtree(cfg.OUTPUT_DIR)
        print(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = MyTrainer(cfg) #DefaultTrainer(cfg)  #Trainer(cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        
    def run_train(self, train_json, img_dir_train, dataset_name, val_json, img_dir_val):
        self.reset(train_json, img_dir_train, dataset_name)
        # self.vis()
        self.train(val_json, img_dir_val)


maxiters = [500]
lrs = [0.001, 0.002]
warmups = [100]

# # maxiters = [1000, 2000]
# maxiters = [500, 1000, 2000, 4000, 6000]
# # lrs = [0.0001, 0.0005, 0.001, 0.002, 0.005]
# lrs = [0.001, 0.0005, 0.002]
# warmups = [100, 200]

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

def run_training(out_dir, img_dir_train, n=10, active=False):
    train_json = os.path.join(out_dir, 'coco_train.json')
    for lr in lrs:
        for warmup in warmups:
            for maxiter in maxiters:
                results = {
                    "bbox": {
                        "AP50": []
                    },
                    "segm": {
                        "AP50": []
                    }
                }
                for i in range(n):
                    dataset_name = "/".join(out_dir.split('/')[-3 if active else -2:])
                    print(f'dataset_name {dataset_name}')
                    c = COCOTrain(lr, warmup, maxiter, i, dataset_name)
                    print(f'dataset_name {dataset_name}')
                    c.run_train(train_json, img_dir_train, dataset_name, val_json0, img_dir_val0)