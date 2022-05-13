# get candidate image_ids, spawn location, target xyz
from droidlet.perception.robot.active_vision.candidate_selection import SampleGoodCandidates
from droidlet.lowlevel.robot_coordinate_utils import xyz_pyrobot_to_canonical_coords, base_pyrobot_coords_to_canonical_coords
from droidlet.perception.robot.handlers import convert_depth_to_pcd, compute_uvone
from common_utils import is_annot_validfn_class, is_annot_validfn_inst, log_time, class_labels, instance_ids
from typing import List

import os 
import glob
import json
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import argparse
import submitit
import shutil

def visualize_instances(traj_path, out_dir, candidates):
    vis_path = os.path.join(out_dir, 'candidate_selection_visuals')
    os.makedirs(vis_path, exist_ok=True)
    for img_indx, label in candidates:
        src_img = cv2.imread(os.path.join(traj_path, "rgb/{:05d}.jpg".format(img_indx)))
        src_label = np.load(os.path.join(traj_path, "seg/{:05d}.npy".format(img_indx)))
        all_label = np.bitwise_or(src_label == label, np.zeros_like(src_label))
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(src_img)
        ax2.imshow(all_label)
        ax2.set_title(f'label {label}')
        plt.savefig(f"{vis_path}/{img_indx:05d}.jpg")
        # plt.show()

def get_center(mask):
    m = np.argwhere(mask == 1)
    return m[int(len(m)/2)]

def get_center_of_target_label(seg, label):
    return get_center(seg == label), seg == label

def save_mask_pcd(pts, mask):
    # Save pcd for mask
    pass

def get_target(traj_path, img_indx, target_label, base_pos):
    src_depth = np.load(os.path.join(traj_path, "depth/{:05d}.npy".format(img_indx)))
    src_pcd = np.load(os.path.join(traj_path, "pcd/{:05d}.npy".format(img_indx)))
    src_label = np.load(os.path.join(traj_path, "seg/{:05d}.npy".format(img_indx)))
    
    # pick the larger mask, get centroid for that 
    c, _ = get_center_of_target_label(src_label, target_label)
    if c is None:
        return None
        
    height, width = src_depth.shape
    # uv_one_in_cam, intrinsic_mat, rot, trans = compute_uvone(height, width)
    # pts = convert_depth_to_pcd(src_depth, base_pos, uv_one_in_cam, rot, trans)
    pts = src_pcd.reshape((height, width, 3))
    
    # TODO: visualize pts corresponding to mask 
    # save_mask_pcd(pts, mask)

    xyz_pyrobot = pts[c[0],c[1]]
    return xyz_pyrobot_to_canonical_coords(xyz_pyrobot).tolist()

    
def process(traj_path, out_dir, gt, s, is_annot_validfn):
    # src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
    candidates = s.get_n_good_candidates_across_all_labels(gt)
    print(f'candidates {candidates}')
    
    reexplore_task_data = {}
    
    base_poses = []
    base_poses_hab = []
    if os.path.isfile(os.path.join(traj_path, 'data_hab.json')):
        with open(os.path.join(traj_path, 'data_hab.json'), 'r') as f:
            data = json.load(f)
            for x, _ in candidates:
                base_poses_hab.append(data[str(x)])
                
    if os.path.isfile(os.path.join(traj_path, 'data.json')):
        with open(os.path.join(traj_path, 'data.json'), 'r') as f:
            data = json.load(f)
            for x, _ in candidates:
                base_poses.append(data[str(x)])
    
    # get target xyz
    target_xyz = []
    for i in range(len(candidates)):
        target_xyz.append(get_target(traj_path, candidates[i][0], candidates[i][1], base_poses[i]))
        visualize_instances(traj_path, out_dir, candidates)
        reexplore_task_data[i] = {
            'src_img_id': int(candidates[i][0]),
            'spawn_pos': base_pyrobot_coords_to_canonical_coords(base_poses[i]),
            'base_pos': base_pyrobot_coords_to_canonical_coords(base_poses[i]),
            'target': target_xyz[i],
            'label': int(candidates[i][1]),
        }
        
    with open(os.path.join(out_dir, 'reexplore_data.json'), 'w') as f:
        json.dump(reexplore_task_data, f)

    with open(os.path.join(out_dir, 'traj_path.txt'), 'w') as f:
        f.write(traj_path)

def get_trajectory_size(traj_path: str) -> int:
    """Returns the size of the trajectory.
    Size is defined as the number of images in a trajectory
    """
    img_dir = os.path.join(traj_path, 'rgb')
    if not os.path.isdir(img_dir):
        return -1
    return len(glob.glob(img_dir + '/*.jpg'))

def get_valid_trajectories(baseline_root: str) -> List[str]:
    """Selects trajectories with atleast a 100 frames"""
    valid_trajs = []
    for traj_path in glob.glob(baseline_root + '/*'):
        if traj_path.split('/')[-1].isdigit():
            if get_trajectory_size(traj_path) > 100:
                valid_trajs.append(traj_path)
    print(f'{len(valid_trajs)} valid trajectories!')
    return valid_trajs

def process_robot(root_dir, out_dir):
    seg_dir = os.path.join(root_dir, 'seg')
    #FIXME: Read actual labels
    candidates = [(int(x.split('.')[0]), 1) for x in os.listdir(seg_dir)]

    print(f'candidates {candidates}')
    reexplore_task_data = {}
    base_poses = []

    if os.path.isfile(os.path.join(root_dir, 'data.json')):
        with open(os.path.join(root_dir, 'data.json'), 'r') as f:
            data = json.load(f)
            for x, _ in candidates:
                base_poses.append(data[str(x)])
    
    target_xyz = []
    for i in range(len(candidates)):
        target_xyz.append(get_target(root_dir, candidates[i][0], candidates[i][1], base_poses[i]))
        visualize_instances(root_dir, out_dir, candidates)
        reexplore_task_data[i] = {
            'src_img_id': int(candidates[i][0]),
            'spawn_pos': base_pyrobot_coords_to_canonical_coords(base_poses[i]),
            'base_pos': base_pyrobot_coords_to_canonical_coords(base_poses[i]),
            'target': target_xyz[i],
            'label': int(candidates[i][1]),
        }
    
    with open(os.path.join(out_dir, 'reexplore_data.json'), 'w') as f:
        json.dump(reexplore_task_data, f)


def find_spawn_loc(
        baseline_root: str, 
        out_dir: str, 
        num_traj: int, 
        job_dir: str,
        mode: str,
        setting: str,
    ) -> None:
    """
    Main fn to find the spawn locations for reexplore for all trajectories in baseline_root
    """
    jobs = []
    print(f"baseline_root {baseline_root}")
    if mode == 'robot':
        process_robot(baseline_root, out_dir)

    elif mode == 'sim':
        with executor.batch():
            for traj_path in get_valid_trajectories(baseline_root)[:num_traj]:
                if traj_path.split('/')[-1].isdigit():
                    print(f'processing {traj_path}')
                    traj_id = '/'.join(traj_path.split('/')[-2:])
                    for setting in [setting]:
                        annot_fn = is_annot_validfn_class if setting == 'class' else is_annot_validfn_inst
                        labels = class_labels if setting == 'class' else instance_ids

                        @log_time(os.path.join(job_dir, 'job_log.txt'))
                        def job_unit(traj_path, out_dir, traj_id, annot_fn, labels, setting):
                            s = SampleGoodCandidates(traj_path, annot_fn, labels, setting)
                            for gt in range(5,15,4):
                                outr = os.path.join(out_dir, traj_id, setting, str(gt))
                                os.makedirs(outr, exist_ok=True)
                                print(f'outr {outr}')
                                process(traj_path, outr, gt, s, annot_fn)

                        # job_unit(traj_path, out_dir, traj_id, annot_fn, labels, setting)
                        job = executor.submit(job_unit, traj_path, out_dir, traj_id, annot_fn, labels, setting)
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
    parser.add_argument("--out_dir", type=str, default="", help="")
    parser.add_argument("--job_dir", type=str, default="", help="")
    parser.add_argument("--setting", type=str)
    parser.add_argument("--num_traj", type=int, default=-1)
    parser.add_argument("--mode", 
        type=str, 
        default="sim", 
        help="two modes: sim (runs on slurm) or robot (runs locally)"
    )

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="devlab", #"learnfair", #scavenge
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

    # Ensure clean out_dir
    if os.path.isdir(args.out_dir):
        print(f'rmtree {args.out_dir}')
        shutil.rmtree(args.out_dir)

    find_spawn_loc(
        args.data_dir, args.out_dir, args.num_traj, args.job_dir, args.mode, args.setting
    )