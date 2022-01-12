import argparse
import submitit
import glob
from pathlib import Path
import os
import subprocess
import json
import logging
from shutil import copyfile, rmtree
import ray
import numpy as np
import cv2
from droidlet.perception.robot import LabelPropagate
from PIL import Image
import matplotlib.pyplot as plt
from coco import run_coco
from train import run_training

combinations = {
    'e1r1r2': ['e1', 'r1', 'r2'],
    'e1s1r2': ['e1', 's1', 'r2'],
    'e1c1sr2': ['e1', 'c1s', 'r2'],
    'e1c1lr2': ['e1', 'c1l', 'r2'],
    'e1s1c1s': ['e1', 's1', 'c1s'],
    'e1s1c1l': ['e1', 's1', 'c1l'],
}

def get_training_data(path):
    """
    Returns a struct for train, val, test data and associated directories
    """
    print(f'\nget_training_data {path}')
    if 'instance' in path:
        return {
            'out_dir': path,
            'dataset_name': path.split('/')[-1],
            'train': {
                'json': os.path.join(path, 'coco_train.json'), 
                'img_dir': os.path.join(path, 'rgb')
            },
            'val': {
                'json': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/coco_val.json',
                'img_dir': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/rgb',
            },
        }

    elif 'class' in path:
        return {
            'out_dir': path,
            'dataset_name': path.split('/')[-1],
            'train': {
                'json': os.path.join(path, 'coco_train.json'),
                'img_dir': os.path.join(path, 'rgb'),
            },
            'val': {
                'json': '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_train.json',
                'img_dir': '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/rgb',
            },
            'test':{
                'json': '/checkpoint/apratik/finals/jsons/active_vision/frlapt1_20n0.json',
                'img_dir': '/checkpoint/apratik/ActiveVision/active_vision/replica_random_exploration_data/frl_apartment_1/rgb',
            },
        }

    return None

def prep_and_run_training(data_dir, job_dir, args):
    print(f'preparing and running training for {data_dir}')
    jobs = []

    with executor.batch():
        for path in Path(data_dir).rglob('pred_label*'):
            path = str(path)
            for k in combinations.keys():
                if k in path:
                    def job_unit(path, args):
                        run_coco(path)
                        training_data = get_training_data(path)
                        run_training(training_data, args.num_train_samples)
                    print(f'launching training for {path}')
                    job = executor.submit(job_unit, path, args)
                    jobs.append(job)

    if len(jobs) > 0:
        print(f"Job Id {jobs[0].job_id.split('_')[0]}, num jobs {len(jobs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_dir",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument("--job_dir", type=str, default="", help="")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")
    parser.add_argument("--num_train_samples", type=int, default=1, help="total number of times we want to train the same model")

    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=1000,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1, 
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="Droidlet Active Vision Pipeline"
    )

    prep_and_run_training(args.data_dir, args.job_dir, args)