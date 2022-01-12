import argparse
import submitit
import glob
from pathlib import Path
import os
import subprocess
import json
import logging

def launch_reexplore(data_dir, job_dir, noise):
    # find all folders with reexplore_data.json
    print(f'data_dir {data_dir}')
    jobs = []

    with executor.batch():
        for path in Path(data_dir).rglob('reexplore_data.json'):
            with open(os.path.join(path.parent, 'reexplore_data.json'), 'r') as f:
                data = json.load(f)
                if len(data.keys()) > 0:
                    print(f'processing {path.parent}')
                    def job_unit(p, noise):
                        comm = f"./launch_reexplore.sh {p} {p}/reexplore_data.json {noise}"
                        print(f'command {comm}')
                        process = subprocess.Popen(comm.split(), stdout=subprocess.PIPE, cwd='/private/home/apratik/fairo')
                        output, error = process.communicate()
                        logging.info(f'output {output} error {error}')

                    # job_unit(path.parent, noise)
                    job = executor.submit(job_unit, path.parent, noise)
                    jobs.append(job)

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

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
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

    launch_reexplore(args.data_dir, args.job_dir, args.noise)