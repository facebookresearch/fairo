import argparse
import submitit
from pathlib import Path
import os
import subprocess
import json
import logging
from typing import List
from common_utils import log_time

def get_valid_trajectories(data_dir: str) -> List[str]:
    """
    Valid trajectories have non-empty reexploration data
    """
    valid = set()
    for path in Path(data_dir).rglob('reexplore_data.json'):
        with open(os.path.join(path.parent, 'reexplore_data.json'), 'r') as f:
            data = json.load(f)
            if len(data.keys()) > 0:
                valid.add(path.parent.parent.parent)
    
    return list(valid)

class reexplore_job:
    def __init__(self):
        pass 

    def __call__(self, p, noise):
        comm = f"./reexplore.sh {p} {p}/reexplore_data.json {noise}"
        logging.info(f'command {comm}')
        process = subprocess.Popen(comm.split(), stdout=subprocess.PIPE, cwd='.')
        output, error = process.communicate()
        logging.info(f'output {output} error {error}')

    def checkpoint(self, p, noise) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, p, noise)  # submits to requeuing

def launch_reexplore(data_dir: str, job_dir: str, noise: bool, num_traj: int) -> None:
    """
    Launches reexplore for every reexplore_data.json 
    (the json is output by the ./launch_candidate_selection.sh script)
    """
    jobs = []

    valid_trajs = get_valid_trajectories(data_dir)
    print(f'{len(valid_trajs)} valid trajectories!')

    with executor.batch():
        for traj_dir in valid_trajs:
            for path in Path(traj_dir).rglob('reexplore_data.json'):
                with open(os.path.join(path.parent, 'reexplore_data.json'), 'r') as f:
                    data = json.load(f)
                    if len(data.keys()) > 0 and 'instance/5' in str(path):
                        print(f'processing {path.parent}')
                        reexplore_callable = reexplore_job()
                        reexplore_callable(path.parent, noise)
                        # job = executor.submit(reexplore_callable, path.parent, noise)
                        # jobs.append(job)

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
        timeout_min=120,
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

    launch_reexplore(args.data_dir, args.job_dir, args.noise, args.num_traj)