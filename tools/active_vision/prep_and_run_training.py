import argparse
import submitit
from pathlib import Path
import os
from coco import run_coco
from train import run_training
from common_utils import log_time, heuristics as heus

combinations = {
    'r1r2': ['r1', 'r2'],
    's1ppr2': ['s1pp', 'r2'],
    'c1ppr2': ['c1pp', 'r2'],
    'c1pps1pp': ['c1pp', 's1pp'],
}

# combinations = {
#     'r1': ['r1'],
#     's1pp': ['s1pp'],
#     'c1pp': ['c1pp'],
# }

def get_training_data(path, job_dir):
    """
    Returns a struct for train, val, test data and associated directories
    """
    def get_path(path, job_dir, with_gt = ''):
        suffix = '/'.join(path.split('/')[-6:])
        job_path = os.path.join(job_dir, suffix+'_'+with_gt)
        print(f'job_path {job_path}')
        return job_path

    print(f'\nget_training_data {path}')
    return {
        'out_dir': get_path(path, job_dir),
        'dataset_name': path.split('/')[-1],
        'train': {
            'json': os.path.join(path, 'coco_train.json'), 
            'img_dir': os.path.join(path, 'rgb')
        },
        'val': {
            'json': '/checkpoint/apratik/jobs/reexplore/train_data_1805/train_data_1805/val_set/coco_train.json',
            'img_dir': '/checkpoint/apratik/jobs/reexplore/train_data_1805/train_data_1805/val_set/rgb'
        },
        'test': {
            'json': '/checkpoint/apratik/jobs/reexplore/train_data_1805/train_data_1805/test_set/coco_train.json',
            'img_dir': '/checkpoint/apratik/jobs/reexplore/train_data_1805/train_data_1805/test_set/rgb'
        }
        # 'val': {
        #     'json': '/checkpoint/apratik/jobs/reexplore/train_data_solo2/val_set/coco_val.json',
        #     'img_dir': '/checkpoint/apratik/jobs/reexplore/train_data_solo2/val_set/rgb'
        # },
        # 'test': {
        #     'json': '/checkpoint/apratik/jobs/reexplore/robot_training_data/validation_set/coco_val.json',
        #     'img_dir': '/checkpoint/apratik/jobs/reexplore/robot_training_data/validation_set/rgb'
        # }
    }

def prep_and_run_training(data_dir: str, job_dir: str, num_train_samples: int, setting: str) -> None:
    print(f'preparing and running training for {data_dir}')
    jobs = []
    trajs = set()
    with executor.batch():
        for path in Path(data_dir).rglob('pred_label*'):
            for k in combinations.keys():
                if k in str(path):
                    print(f'launching training for {str(path)} ...')
                    @log_time(os.path.join(job_dir, 'job_log.txt'))
                    def job_unit(path, num_train_samples, job_dir):
                        print(f'run_coco on {path}')
                        run_coco(str(path))
                        training_data = get_training_data(path, job_dir)
                        print(training_data, job_dir)
                        run_training(training_data, num_train_samples)
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
    parser.add_argument("--setting", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")
    parser.add_argument("--num_train_samples", type=int, default=3, help="total number of times we want to train the same model")

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
        slurm_comment="NeurIPS deadline 5/19 Droidlet Active Vision"
    )

    prep_and_run_training(args.data_dir, args.job_dir, args.num_train_samples, args.setting)