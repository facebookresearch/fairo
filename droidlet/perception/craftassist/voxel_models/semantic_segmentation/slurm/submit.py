import os
import argparse
import uuid
from pathlib import Path
import submitit
from train_semantic_segmentation import get_parser, main


class SubmititMain:
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        self.func(args)

    def checkpoint(self, args):
        # bit hacky, but remove dist init file
        os.remove(args.dist_init[7:])
        return submitit.helpers.DelayedSubmission(self, args)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="")
parser.add_argument("--folder", type=str, default="", help="")
parser.add_argument("--partition", type=str, default="learnfair", help="")
parser.add_argument("--ngpus", type=int, default=8, help="")
parser.add_argument("--nodes", type=int, default=1, help="")
parser.add_argument("--constraint", type=str, default="volta32gb", help="")
parser.add_argument("--time", type=int, default=72, help="hours")
parser.add_argument("--args", type=str, default="", help="")
args = parser.parse_args()

job_args = args.args.split()
folder = Path(args.folder)
os.makedirs(str(folder), exist_ok=True)
init_file = folder / f"{uuid.uuid4().hex}_init"  # not used when nodes=1
job_args += ["--dist-init", init_file.as_uri()]
job_parser = get_parser()
job_args = job_parser.parse_args(job_args)

executor = submitit.AutoExecutor(folder=folder / "%j", max_num_timeout=10)
executor.update_parameters(
    mem_gb=256,
    gpus_per_node=args.ngpus,
    tasks_per_node=args.ngpus,  # one task per GPU
    cpus_per_task=8,
    nodes=args.nodes,
    timeout_min=args.time * 60,
    # Below are cluster dependent parameters
    slurm_partition=args.partition,
    signal_delay_s=120,
    constraint=args.constraint,
    additional_parameters={
        "mail-user": f"{os.environ['USER']}@fb.com",
        "mail-type": "fail",
    },
)
if args.partition == "priority":
    executor.update_parameters(slurm_comment="ICLR 2023 Sep 28")

executor.update_parameters(name=args.name)
submitit_main = SubmititMain(main)
job = executor.submit(submitit_main, job_args)
print("submited {} {}".format(job.job_id, args.name))