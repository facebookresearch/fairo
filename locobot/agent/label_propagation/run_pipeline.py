from label_propagation import run_label_prop
from coco import run_coco
from slurm_train import run_training
import submitit
import argparse
import os
import glob
from datetime import datetime

def _runner(traj, gt, p, args):
    start = datetime.now()
    traj_path = os.path.join(args.data_path, str(traj))
    if os.path.isdir(traj_path):
        outdir = os.path.join(args.job_folder, str(traj), f'pred_label_gt{gt}p{p}')
        run_label_prop(outdir, gt, p, traj_path)
        if len(glob.glob1(outdir,"*.npy")) > 0:
            run_coco(outdir, traj_path)
            run_training(outdir, os.path.join(traj_path, 'rgb'), args.num_train_samples)
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
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--num_traj", type=int, default=1, help="total number of trajectories to run pipeline for")
    parser.add_argument("--num_train_samples", type=int, default=1, help="total number of times we want to train the same model")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_folder, 'slurm_logs'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="prioritylab", #scavenge
        timeout_min=2000,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1, 
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="ICRA 2022 deadline, 9/14"
    )

    jobs = []
    if args.slurm:
        with executor.batch():
            for traj in range(args.num_traj):
                for gt in range(5, 30, 5):
                    for p in range(2, 10, 2):
                        job = executor.submit(_runner, traj, gt, p, args)
                        jobs.append(job)
        
        print(f'{len(jobs)} jobs submitted')
    
    else:
        for traj in range(args.num_traj):
                for gt in range(5, 30, 5):
                    for p in range(2, 4, 2): # only run for fixed p 
                        _runner(traj, gt, p, args)