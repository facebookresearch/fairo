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

def prep_and_run_training(data_dir, job_dir, args):
    print(f'data_dir {data_dir}')
    jobs = []

    with executor.batch():
        for path in Path(data_dir).rglob('pred_label*'):
            path = str(path)
            for k in combinations.keys():
                if k in path:
                    print(path)
                    run_coco(path)
                    run_training(path, os.path.join(path, 'rgb'), args.num_train_samples)
                    break

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

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=30,
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