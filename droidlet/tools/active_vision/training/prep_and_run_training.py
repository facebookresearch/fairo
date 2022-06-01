import argparse
import submitit
from pathlib import Path
import os
from coco import run_coco
from train import run_training
from droidlet.tools.active_vision.common_utils import log_time, heuristics as heus, combinations


def get_training_data(path, job_dir):
    """
    Returns a struct for train, val, test data and associated directories
    """

    def get_path(path, job_dir, with_gt=""):
        suffix = "/".join(path.split("/")[-5:])
        job_path = os.path.join(job_dir, suffix + "_" + with_gt)
        print(f"job_path {job_path}")
        return job_path

    print(f"\nget_training_data {path}")
    if "instance" in path:
        return [
            {
                "out_dir": get_path(path, job_dir),
                "dataset_name": path.split("/")[-1],
                "train": {
                    "json": os.path.join(path, "coco_train.json"),
                    "img_dir": os.path.join(path, "rgb"),
                },
                "val": {
                    "json": "/checkpoint/apratik/jobs/reexplore/instance_val_set_8/coco_val_all.json",
                    "img_dir": "/checkpoint/apratik/jobs/reexplore/instance_val_set_8/rgb",
                },
            },
            {
                "out_dir": get_path(path, job_dir, "with_gt"),
                "dataset_name": path.split("/")[-1],
                "train": {
                    "json": os.path.join(path, "coco_train_with_seg_gt.json"),
                    "img_dir": os.path.join(path, "rgb"),
                },
                "val": {
                    "json": "/checkpoint/apratik/jobs/reexplore/instance_val_set_8/coco_val_all.json",
                    "img_dir": "/checkpoint/apratik/jobs/reexplore/instance_val_set_8/rgb",
                },
            },
        ]

    elif "class" in path:
        return {
            "out_dir": get_path(path, job_dir),
            "dataset_name": path.split("/")[-1],
            "train": {
                "json": os.path.join(path, "coco_train.json"),
                "img_dir": os.path.join(path, "rgb"),
            },
            "val": {
                "json": "/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_train.json",
                "img_dir": "/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/rgb",
            },
            "test": {
                "json": "/checkpoint/apratik/finals/jsons/active_vision/frlapt1_20n0.json",
                "img_dir": "/checkpoint/apratik/ActiveVision/active_vision/replica_random_exploration_data/frl_apartment_1/rgb",
            },
        }

    return None


import functools


@functools.lru_cache(None)
def sanity_check_traj(x, setting):
    heuristics = set(heus)
    is_valid = True
    tid = x.split("/")[-1]
    x = os.path.join(x, setting)
    for gt in os.listdir(x):
        gt_path = os.path.join(x, gt)
        reex_objects = [x for x in os.listdir(gt_path) if x.isdigit()]
        # print(reex_objects)
        # print(f'traj {tid} texpected objects {gt}, actual {len(reex_objects)}')
        if int(gt) != len(reex_objects):
            print(f"only {len(reex_objects)} objects in {gt_path}")
            is_valid = False

        # now check each subfolder
        for sub in reex_objects:
            # each should have all the heuristics
            sub_path = os.path.join(gt_path, sub)
            hf = set(os.listdir(sub_path))
            if hf != heuristics:
                print(f"only {hf} heuristics found in {sub_path}!!")
                is_valid = False
            # else:
            # print(f'all {hf} heuristics found in {sub_path}!!')
    return is_valid


def prep_and_run_training(
    data_dir: str, job_dir: str, num_train_samples: int, setting: str
) -> None:
    print(f"preparing and running training for {data_dir}")
    jobs = []
    trajs = set()
    with executor.batch():
        for traj_id in os.listdir(data_dir):
            traj_dir = os.path.join(data_dir, traj_id)
            if traj_id.isdigit() and sanity_check_traj(traj_dir, setting):
                trajs.add(traj_id)
                print(f"traj_id {traj_id}, traj_dir {traj_dir}")
                for path in Path(traj_dir).rglob("pred_label*"):
                    for k in combinations.keys():
                        if k in str(path):
                            print(f"launching training for {str(path)} ...")

                            @log_time(os.path.join(job_dir, "job_log.txt"))
                            def job_unit(path, num_train_samples, job_dir):
                                run_coco(path)
                                training_data = get_training_data(path, job_dir)
                                print(training_data, job_dir)
                                for td in training_data:
                                    run_training(td, num_train_samples)

                            print(f"launching training for {path}")
                            # job_unit(str(path), num_train_samples, job_dir)
                            job = executor.submit(job_unit, str(path), num_train_samples, job_dir)
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
    parser.add_argument("--setting", type=str, help="instance or class")
    parser.add_argument(
        "--slurm",
        action="store_true",
        default=False,
        help="Run the pipeline on slurm, else locally",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=1,
        help="total number of times we want to train the same model",
    )

    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, "slurm_logs/%j"))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair",  # "learnfair", #scavenge
        timeout_min=100,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1,
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="NeurIPS deadline 5/19 Droidlet Active Vision",
    )

    prep_and_run_training(args.data_dir, args.job_dir, args.num_train_samples, args.setting)
