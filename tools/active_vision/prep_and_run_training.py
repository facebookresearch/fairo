import argparse
import submitit
from pathlib import Path
import os
from coco import run_coco
from train import run_training
from common_utils import log_time

combinations = {
    'e1r1r2': ['e1', 'r1', 'r2'],
    'e1s1r2': ['e1', 's1', 'r2'],
    'e1c1sr2': ['e1', 'c1s', 'r2'],
    'e1c1lr2': ['e1', 'c1l', 'r2'],
    # 'e1s1c1s': ['e1', 's1', 'c1s'],
    # 'e1s1c1l': ['e1', 's1', 'c1l'],
}

def get_training_data(path, job_dir):
    """
    Returns a struct for train, val, test data and associated directories
    """
    def get_path(path, job_dir, with_gt = ''):
        suffix = '/'.join(path.split('/')[-5:])
        job_path = os.path.join(job_dir, suffix+'_'+with_gt)
        print(f'job_path {job_path}')
        return job_path

    print(f'\nget_training_data {path}')
    if 'instance' in path:
        return [{
            'out_dir': get_path(path, job_dir),
            'dataset_name': path.split('/')[-1],
            'train': {
                'json': os.path.join(path, 'coco_train.json'), 
                'img_dir': os.path.join(path, 'rgb')
            },
            'val': {
                'json': '/checkpoint/apratik/jobs/reexplore/instance_val_set/coco_val_all.json',
                'img_dir': '/checkpoint/apratik/jobs/reexplore/instance_val_set/rgb'
            }
            # 'val': {
            #     'json': '/checkpoint/apratik/jobs/reexplore/instance_val_set_60/coco_val_all.json',
            #     'img_dir': '/checkpoint/apratik/jobs/reexplore/instance_val_set_60/rgb'
            # }
            # 'val': {
            #     'json': '/checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1116_cvpr2/baseline/coco_val_all.json', 
            #     'img_dir': '/checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1116_cvpr2/baseline/rgb'
            # },
            # 'val': {
            #     'json': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/coco_val.json',
            #     'img_dir': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/rgb',
            # },
        },
        {
            'out_dir': get_path(path, job_dir, 'with_gt'),
            'dataset_name': path.split('/')[-1],
            'train': {
                'json': os.path.join(path, 'coco_train_with_seg_gt.json'), 
                'img_dir': os.path.join(path, 'rgb')
            },
            'val': {
                'json': '/checkpoint/apratik/jobs/reexplore/instance_val_set/coco_val_all.json',
                'img_dir': '/checkpoint/apratik/jobs/reexplore/instance_val_set/rgb'
            }
            # 'val': {
            #     'json': '/checkpoint/apratik/jobs/reexplore/instance_val_set_60/coco_val_all.json',
            #     'img_dir': '/checkpoint/apratik/jobs/reexplore/instance_val_set_60/rgb'
            # }
            # 'val': {
            #     'json': '/checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1116_cvpr2/baseline/coco_val_all.json', 
            #     'img_dir': '/checkpoint/apratik/data_devfair0187/apartment_0/instance_det_sampled1_1116_cvpr2/baseline/rgb'
            # },
            # 'val': {
            #     'json': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/coco_val.json',
            #     'img_dir': '/checkpoint/apratik/jobs/active_vision/pipeline/instance_det/apartment_0/test_1116_cvpr2/rgb',
            # },
        }]

    elif 'class' in path:
        return {
            'out_dir': get_path(path, job_dir),
            'dataset_name': path.split('/')[-1],
            'train': {
                'json': os.path.join(path, 'coco_train.json'),
                'img_dir': os.path.join(path, 'rgb'),
            },
            'val': {
                'json': '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/seg/coco_train.json',
                'img_dir': '/checkpoint/apratik/data/data/apartment_0/default/no_noise/mul_traj_200/83/rgb',
            },
            'test':{
                'json': '/checkpoint/apratik/finals/jsons/active_vision/frlapt1_20n0.json',
                'img_dir': '/checkpoint/apratik/ActiveVision/active_vision/replica_random_exploration_data/frl_apartment_1/rgb',
            },
        }

    return None

def prep_and_run_training(data_dir: str, job_dir: str, num_train_samples: int) -> None:
    print(f'preparing and running training for {data_dir}')
    jobs = []
    # def get_paths(traj_ids, data_dir):
    #     all_paths = [str(path) for path in Path(data_dir).rglob('pred_label*') if '/instance/5/' in str(path)]
    #     paths = []
    #     for x in traj_ids:
    #         lookup = f'/{x}/instance/5/'
    #         paths.append([path for path in all_paths if lookup in path])
    #     return [x for y in paths for x in y]

    # test_paths = get_paths([0,2,6,8,10], data_dir)
    # test_paths = [str(path) for path in Path(data_dir).rglob('pred_label*') if 'baselinev3/8/class/5' in str(path)]
    # print(f'{len(test_paths)} test paths in class')
    with executor.batch():
        # for path in test_paths:
        for path in Path(data_dir).rglob('pred_label*'):
            if any(p in str(path.parent) for p in ['e1r1r2', 'e1s1r2', 'e1c1lr2', 'e1c1sr2']): #if '/instance/5' in str(path.parent):
                for k in combinations.keys():
                    if k in str(path):
                        @log_time(os.path.join(job_dir, 'job_log.txt'))
                        def job_unit(path, num_train_samples, job_dir):
                            run_coco(path)
                            training_data = get_training_data(path, job_dir)
                            print(training_data, job_dir)
                            for td in training_data:
                                run_training(td, num_train_samples)
                        print(f'launching training for {path}')
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
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")
    parser.add_argument("--num_train_samples", type=int, default=4, help="total number of times we want to train the same model")

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
        slurm_comment="Droidlet Active Vision Pipeline"
    )

    prep_and_run_training(args.data_dir, args.job_dir, args.num_train_samples)