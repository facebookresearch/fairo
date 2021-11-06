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

# val_json = '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_train.json'
# val_json = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/seg/coco_valgtfix.json'
# img_dir_val = '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/rgb'
# img_dir_val = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/rgb'

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

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
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
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
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
        self.results = {
            "bbox": {
                "AP50": []
            },
            "segm": {
                "AP50": []
            }
        }
        self.val_results = {
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
#     ['chair', 'door', 'table', 'indoor-plant', 'cushion', 'sofa'] != ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']

            
    def train(self, val_json, img_dir_val):
        cfg = self.cfg
        print(f'SOLVER PARAMS {cfg.SOLVER.MAX_ITER, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.BASE_LR}')
        cfg.DATASETS.TRAIN = (self.train_data,)
        
        self.val_data = self.dataset_name + "_val" + str(self.seed)
        self.val_json = val_json
        cfg.DATASETS.TEST = (self.val_data,self.train_data)
        register_coco_instances(self.val_data, {}, val_json, img_dir_val)
        MetadataCatalog.get(self.val_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        
        cfg.TEST.EVAL_PERIOD = 100
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_yaml)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 16
        
        cfg.SOLVER.GAMMA=0.75
        cfg.SOLVER.STEPS=tuple([200*(i+1) for i in range(100) if 200*(i+1) < cfg.SOLVER.MAX_ITER])
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
        MetadataCatalog.get(self.train_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        print(f'classes {MetadataCatalog.get(self.train_data)}')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(self.train_data).get("thing_classes"))  
        cfg.OUTPUT_DIR = os.path.join('output', self.name, str(cfg.SOLVER.MAX_ITER), str(cfg.SOLVER.BASE_LR), str(cfg.SOLVER.WARMUP_ITERS), self.dataset_name + str(self.seed))
        print(f"recreating {cfg.OUTPUT_DIR}")
        # if os.path.isdir(cfg.OUTPUT_DIR):
        #     shutil.rmtree(cfg.OUTPUT_DIR)
        print(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = MyTrainer(cfg) #DefaultTrainer(cfg)  #Trainer(cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        
    def run_val(self, dataset_name, val_json, img_dir_val):
#         self.val_data = dataset_name + "_val" + str(self.seed)
#         self.val_json = val_json
#         self.cfg.DATASETS.TEST = (self.val_data,)
#         register_coco_instances(self.val_data, {}, val_json, img_dir_val)
#         MetadataCatalog.get(self.val_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
#         print(f'classes {MetadataCatalog.get(self.val_data)}')
        self.evaluator = COCOEvaluator(self.val_data, ("bbox", "segm"), False, output_dir=self.cfg.OUTPUT_DIR)
        self.val_loader = build_detection_test_loader(self.cfg, self.val_data)
        results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
        self.val_results['bbox']['AP50'].append(results['bbox']['AP50'])
        self.val_results['segm']['AP50'].append(results['segm']['AP50'])
        return results

    def run_test(self, dataset_name, test_json, img_dir_test):
        self.test_data = dataset_name + "_test" + str(self.seed)
        self.test_json = test_json
        self.cfg.DATASETS.TEST = (self.test_data,)
        register_coco_instances(self.test_data, {}, test_json, img_dir_test)
        MetadataCatalog.get(self.test_data).thing_classes = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
        print(f'classes {MetadataCatalog.get(self.test_data)}')
        self.evaluator = COCOEvaluator(self.test_data, ("bbox", "segm"), False, output_dir=self.cfg.OUTPUT_DIR)
        self.val_loader = build_detection_test_loader(self.cfg, self.test_data)
        results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
        self.results['bbox']['AP50'].append(results['bbox']['AP50'])
        self.results['segm']['AP50'].append(results['segm']['AP50'])
        return results
        
    def run_train(self, train_json, img_dir_train, dataset_name, val_json, img_dir_val):
        self.reset(train_json, img_dir_train, dataset_name)
        # self.vis()
        self.train(val_json, img_dir_val)


maxiters = [800,1500,2000]
lrs = [0.0001, 0.0005, 0.001, 0.002, 0.005]
warmups = [100, 200]

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

def run_training(img_dir_train, n, traj, x, gt, p, train_json, val_json, job_dir, name):
    results_dir = os.path.join(job_dir, str(traj), x, str(gt), str(p))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
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
                    c = COCOTrain(lr, warmup, maxiter, i, name)
                    dataset_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
                    print(f'dataset_name {dataset_name}')
                    c.run_train(train_json, img_dir_train, dataset_name, val_json, img_dir_train)
                    res_eval = c.run_val(dataset_name, val_json, img_dir_train)
                    with open(os.path.join(results_dir, "validation_results.txt"), "a") as f:
                        f.write(f'val_json {val_json}\n')
                        f.write(f'lr {lr} warmup {warmup} maxiter {maxiter}\n')
                        f.write(json.dumps(res_eval) + '\n')


data_path = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019'
job_dir = '/checkpoint/apratik/jobs/active_vision/pipeline/apartment_0/straightline/no_noise/04-11-2021/14:30:11'


p0_train = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/seg/coco_gt5_0train.json'
p4_train = '/checkpoint/apratik/jobs/active_vision/pipeline/apartment_0/straightline/no_noise/04-11-2021/14:30:11/1/default/pred_label_gt5p4/coco_train.json'
pr_val = '/checkpoint/apratik/data_devfair0187/apartment_0/straightline/no_noise/1633991019/1/default/seg/coco_gt5_4val.json'

gt = 5
x = 'default'
traj = 1
traj_path = os.path.join(data_path, str(traj), x)

import submitit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision sanity check")
    parser.add_argument("--job_folder", type=str, default="", help="")
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_folder, 'slurm_logs'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="prioritylab", #"learnfair", #scavenge
        timeout_min=2000,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1, 
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="CVPR deadline, 11/16"
    )
    if args.slurm:
        job_baseline = executor.submit(
            run_training, os.path.join(traj_path, 'rgb'), 1, str(traj), x, gt, 0, p0_train, pr_val, args.job_folder, 'p0'
        )
        print(f'job_baseline id {job_baseline.job_id}')

        job_prop = executor.submit(
            run_training, os.path.join(traj_path, 'rgb'), 1, str(traj), x, gt, 4, p4_train, pr_val, args.job_folder, 'p4'
        )
        print(f'job_prop id {job_prop.job_id}')

    else:
        # just sanity checking for errors
        run_training(os.path.join(traj_path, 'rgb'), 1, str(traj), x, gt, 0, p0_train, pr_val, args.job_folder, 'p0')

    

