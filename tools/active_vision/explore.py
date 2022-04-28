import argparse
import submitit
from pathlib import Path
import os
import subprocess
import json
import logging
from typing import List
from common_utils import log_time

class explore_job:
    def __init__(self):
        pass 

    def __call__(self, p, noise):
        comm = f"./explore.sh {p} {noise}"
        print(f'command {comm}')
        process = subprocess.Popen(comm.split(), stdout=subprocess.PIPE, cwd='/private/home/apratik/fairo/tools/active_vision')
        output, error = process.communicate()
        logging.info(f'output {output} error {error}')

    def checkpoint(self, p, noise) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, p, noise)  # submits to requeuing

def launch_explore(args) -> None:
    """
    Launches explore to collect num_traj trajectories
    """
    jobs = []

    with executor.batch():
        for traj_id in range(args.num_traj):
            # data_store_path, noise
            data_store_path = os.path.join(args.data_dir, str(traj_id))
            print(f'data_store_path for traj {traj_id} = {data_store_path}')
            explore_callable = explore_job()
            job = executor.submit(explore_callable, data_store_path, args.noise)
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
    parser.add_argument("--num_traj", type=int, default=2, help="number of trajectories to run reexploration for")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #scavenge
        timeout_min=20,
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

    launch_explore(args)