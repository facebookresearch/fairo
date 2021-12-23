from label_propagation import run_label_prop
from coco import run_coco
from slurm_train import run_training
from droidlet.perception.robot.active_vision.candidate_selection import SampleGoodCandidates
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
from candidates import SampleGoodCandidates

from datetime import datetime
import json

instance_ids = [243,404,196,133,166,170,172]

def is_annot_validfn(annot):
    if annot not in instance_ids:
        return False
    return True

def basic_sanity(traj_path):
    def ll(x, y, ext):
        return len(glob.glob(os.path.join(x, y, f'*{ext}')))
    # load json
    with open(os.path.join(traj_path, 'data.json'), 'r') as f:
        data = json.load(f)
    assert ll(traj_path, 'rgb', '.jpg') == ll(traj_path, 'seg', '.npy') == ll(traj_path, 'depth', '.npy') == len(data.keys())

def log_job_start(args, jobs):
    with open(f"/checkpoint/{os.environ.get('USER')}/jobs/active_vision/pipeline/dec_instance_slurm_launch_start.txt", 'a') as f:
        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment comment {args.comment}\n")
        f.write(f"Num traj {str(args.num_traj)}\n")
        f.write(f"Num train samples {str(args.num_train_samples)}\n")
        f.write(f"Job Folder {args.job_folder}\n")
        f.write(f"Data Dir {args.data_path}\n")
        f.write(f"job_id prefix {str(jobs[0].job_id.split('_')[0])}\n")
        f.write(f"num_jobs {str(len(jobs))}\n")

def logtime(outdir, s):
    start = datetime.now()
    with open(os.path.join(outdir, 'timelog.txt'), "a") as f:
        f.write(f"{start.strftime('%H:%M:%S')}: {s}\n")

def _runner(traj, gt, p, active, data_path, job_folder, num_train_samples, src_img_ids):
    start = datetime.now()
    traj_path = os.path.join(data_path, str(traj))
    print(f'traj_path {traj_path}')
    if os.path.isdir(traj_path):
        basic_sanity(traj_path)
        outdir = os.path.join(job_folder, str(traj), f'pred_label_gt{gt}p{p}')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        logtime(outdir, 'starting..')
        # TODO: use candidate selection
        # src_img_ids = s.get_n_candidates(gt, True)
        logtime(outdir, f'candidate selection done {src_img_ids}')
        # src_img_ids = [10, 20, 30, 40, 50]
        print(f'src_img_ids {src_img_ids}, outdir {outdir}')
        run_label_prop(outdir, gt, p, traj_path, src_img_ids)
        logtime(outdir, 'label prop done')
        if len(glob.glob1(os.path.join(outdir, 'seg'),"*.npy")) > 0:
            # sanity checking coco_val
            run_coco(traj_path, instance_ids, src_img_ids, p, 'coco_val.json', outdir)
            # coco train
            # run_coco(outdir, instance_ids, [], 0, 'coco_train.json', outdir)
            logtime(outdir, 'run coco done')

            # run sanity check
            # run_training(
            #     outdir, 
            #     os.path.join(outdir, 'rgb'), 
            #     os.path.join(outdir, 'coco_train.json'), 
            #     os.path.join(traj_path, 'rgb'),
            #     os.path.join(outdir, 'coco_val.json'), 
            #     1
            # )
            # logtime(outdir, 'Sanity checking done')

            run_training(
                outdir, os.path.join(outdir, 'rgb'), os.path.join(outdir, 'coco_train.json'), None, None, num_train_samples)
            logtime(outdir, 'training done')

            end = datetime.now()
            with open(os.path.join(job_folder, 'timelog.txt'), "a") as f:
                f.write(f"traj {traj}, gt {gt}, p {p} = {(end-start).total_seconds()} seconds, start {start.strftime('%H:%M:%S')}, end {end.strftime('%H:%M:%S')}\n")
    
if __name__ == "__main__":
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
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=2000,
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

    gtps = set()
    for gt in range(5, 15, 5):
        for p in range(0, 30, 5):
            gtps.add((gt,p))

    # for gt in range(5, 30, 5):
    #     for p in range(0,20,5):
    #         gtps.add((gt,p))

    gtps = sorted(list(gtps))
    print(len(gtps), gtps)

    jobs = []
    if args.slurm:
        with executor.batch():
            for traj in range(args.num_traj+1):
                traj_path = os.path.join(args.data_path, str(traj))
                s = SampleGoodCandidates(traj_path, is_annot_validfn)
                for gt, p in gtps: 
                    src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
                    if src_img_ids is not None and len(src_img_ids) > 0:
                        job = executor.submit(
                            _runner, traj, gt, p, args.active, args.data_path, args.job_folder, args.num_train_samples, src_img_ids
                        )
                        jobs.append(job)
        log_job_start(args, jobs)
        print(f'{len(jobs)} jobs submitted')
    
    else:
        print('running locally ...')
        for traj in range(args.num_traj+1):
            traj_path = os.path.join(args.data_path, str(traj))
            s = SampleGoodCandidates(traj_path, is_annot_validfn)
            for gt in range(5, 10, 5):
                for p in range(5, 10, 5): # only run for fixed gt locally to test
                    src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
                    _runner(traj, gt, p, args.active, args.data_path, args.job_folder, args.num_train_samples, src_img_ids)