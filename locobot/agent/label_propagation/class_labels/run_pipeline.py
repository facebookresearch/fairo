from label_propagation import run_label_prop
from coco import run_coco
from slurm_train import run_training
from candidates import PickGoodCandidates
import submitit
import argparse
import os
import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
from PIL import Image
import json
from pycococreatortools import pycococreatortools

from datetime import datetime

def basic_sanity(traj_path):
    def ll(x, y, ext):
        return len(glob.glob(os.path.join(x, y, f'*{ext}')))
    # load json
    with open(os.path.join(traj_path, 'data.json'), 'r') as f:
        data = json.load(f)
    assert ll(traj_path, 'rgb', '.jpg') == ll(traj_path, 'seg', '.npy') == ll(traj_path, 'depth', '.npy') == len(data.keys())

def get_src_img_ids(heu, traj):
    if heu == 'active':
        if traj == 1:
            # return [303, 606, 909, 1212, 1515] # baseline
            return [303, 606, 894, 1212, 1515] # just 894 is the changed frame, partial chair
            # return [355, 606, 894, 1212, 1485] # manually changed 1,3,5th frames
        if traj == 2:
            return [335, 670, 1005, 1340, 1675] #default
            # return [335, 644, 1005, 1340, 1675] # one frame changed
    return 

def log_job_start(args, jobs):
    with open(f"/checkpoint/aszlam/jobs/active_vision/pipeline/class_slurm_launch_start.txt", 'a') as f:
        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment comment {args.comment}\n")
        f.write(f"Num traj {str(args.num_traj)}\n")
        f.write(f"Num train samples {str(args.num_train_samples)}\n")
        f.write(f"Job Folder {args.job_folder}\n")
        f.write(f"Data Dir {args.data_path}\n")
        f.write(f"job_id prefix {str(jobs[0].job_id.split('_')[0])}\n")
        f.write(f"num_jobs {str(len(jobs))}\n")


def _runner(traj, gt, p, args):
    start = datetime.now()
    if not args.active:
        traj_path = os.path.join(args.data_path, str(traj))
        print(f'traj_path {traj_path}')
        if os.path.isdir(traj_path):
            basic_sanity(traj_path)
            outdir = os.path.join(args.job_folder, str(traj), f'pred_label_gt{gt}p{p}')
            s = PickGoodCandidates(traj_path, active=False)
            src_img_ids = s.sample_uniform_nn2(gt)
            # src_img_ids = [10, 20, 30, 40, 50]
            print(f'src_img_ids {src_img_ids}, outdir {outdir}')
            run_label_prop(outdir, gt, p, traj_path, src_img_ids)
            if len(glob.glob1(os.path.join(outdir, 'seg'),"*.npy")) > 0:
                run_coco(outdir)
                run_training(outdir, os.path.join(outdir, 'rgb'), args.num_train_samples)
                end = datetime.now()
                with open(os.path.join(args.job_folder, 'timelog.txt'), "a") as f:
                    f.write(f"traj {traj}, gt {gt}, p {p} = {(end-start).total_seconds()} seconds, start {start.strftime('%H:%M:%S')}, end {end.strftime('%H:%M:%S')}\n")
    else:
        for x in ['default']: #, 'activeonly']:
            traj_path = os.path.join(args.data_path, str(traj), x)
            if os.path.isdir(traj_path):
                basic_sanity(traj_path)
                outdir = os.path.join(args.job_folder, str(traj), x, f'pred_label_gt{gt}p{p}')
                s = PickGoodCandidates(traj_path, active=True)
                src_img_ids = s.sample_uniform_nn2(gt)
                # src_img_ids = get_src_img_ids('active', traj)
                run_label_prop(outdir, gt, p, traj_path, src_img_ids)
                if len(glob.glob1(os.path.join(outdir, 'seg'),"*.npy")) > 0:
                    # run_coco(outdir)
                    # run_training(outdir, os.path.join(outdir, 'rgb'), args.num_train_samples, active=True)
                    end = datetime.now()
                    with open(os.path.join(args.job_folder, 'timelog.txt'), "a") as f:
                        f.write(f"traj {traj}, gt {gt}, p {p} = {(end-start).total_seconds()} seconds, start {start.strftime('%H:%M:%S')}, end {end.strftime('%H:%M:%S')}\n")

if __name__ == "__main__":
    """
    \scene_path
    .\1
    .\2 
    ...

    Outputs propagated semantic labels in out_dir 
    \out_dir (in the format exp_prefix_dt)
    .\slurm_logs
    .\scene_path
    ..\1
    ...\pred_label_gtxpy (labelprop, then coco, then trainig)
    ....\*.npy
    ....\coco_train.json
    ....\train_results.txt 
    ....\label_prop_metric.txt
    ..\2 

    Graphs I want to output
    * active vs baseline vs random
    * label prop metric
    * category wise AP metrics
    """

    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_path",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument("--job_folder", type=str, default="", help="")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--active", action="store_true", default=False, help="Active Setting")
    parser.add_argument("--num_traj", type=int, default=1, help="total number of trajectories to run pipeline for")
    parser.add_argument("--num_train_samples", type=int, default=1, help="total number of times we want to train the same model")

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

    gtps = set()
    for gt in range(5, 15, 5):
        for p in range(0, 30, 5):
            gtps.add((gt,p))

    for gt in range(5, 30, 5):
        for p in range(0,20,5):
            gtps.add((gt,p))

    gtps = sorted(list(gtps))
    print(len(gtps), gtps)

    jobs = []
    if args.slurm:
        with executor.batch():
            for traj in range(args.num_traj+1):
                for gt, p in gtps:
                    job = executor.submit(_runner, traj, gt, p, args)
                    jobs.append(job)
        log_job_start(args, jobs)
        print(f'{len(jobs)} jobs submitted')
    
    else:
        print('running locally ...')
        for traj in range(args.num_traj+1):
                for gt in range(5, 10, 5):
                    for p in range(5, 10, 5): # only run for fixed gt locally to test
                        _runner(traj, gt, p, args)