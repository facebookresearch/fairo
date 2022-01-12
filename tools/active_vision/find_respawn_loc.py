# get candidate image_ids, spawn location, target xyz
from droidlet.perception.robot.active_vision.candidate_selection import SampleGoodCandidates
from droidlet.lowlevel.robot_mover_utils import transform_pose
from droidlet.lowlevel.robot_coordinate_utils import base_canonical_coords_to_pyrobot_coords, xyz_pyrobot_to_canonical_coords
from droidlet.perception.robot.handlers import convert_depth_to_pcd, compute_uvone

import os 
import glob
import json
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import copy
import argparse
import submitit

def visualize_instances(traj_path, img_ids, is_annot_validfn):
    for img_indx in img_ids:
        src_img = cv2.imread(os.path.join(traj_path, "rgb/{:05d}.jpg".format(img_indx)))
        # src_depth = np.load(os.path.join(root, "depth/{:05d}.npy".format(img_indx)))
        src_label = np.load(os.path.join(traj_path, "seg/{:05d}.npy".format(img_indx)))
        all_label = np.zeros_like(src_label).astype(np.int32)
        for x in np.unique(src_label):
            if is_annot_validfn(x):
                all_label = np.bitwise_or(src_label == x, all_label)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(src_img)
        ax2.imshow(all_label)
        plt.show()

def is_annot_validfn_inst(annot):
    instance_ids = [243,404,196,133,166,170,172]
    if annot not in instance_ids:
        return False
    return True

def is_annot_validfn_class(annot):
    def load_semantic_json(scene):
        habitat_semantic_json = f'/checkpoint/apratik/replica/{scene}/habitat/info_semantic.json'
        with open(habitat_semantic_json, "r") as f:
            hsd = json.load(f)
        if hsd is None:
            print("Semantic json not found!")
        return hsd
    hsd = load_semantic_json('apartment_0')
    labels = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
    label_id_dict = {}
    for obj_cls in hsd["classes"]:
        if obj_cls["name"] in labels:
            label_id_dict[obj_cls["id"]] = obj_cls["name"]
    if hsd["id_to_label"][annot] < 1 or hsd["id_to_label"][annot] not in label_id_dict.keys():
        return False
    return True

def get_center(mask):
    # plt.imshow(mask)
    # plt.show()
    m = np.argwhere(mask == 1)
    l = len(m)
    c = m[int(l/2)]
    # print(f'returning center .. {c}')
    return c

def get_center_of_larger_mask(seg, is_annot_validfn):
    larea = 0
    lm = None
    for x in np.unique(seg):
        if is_annot_validfn(x):
            bm = (seg == x)
            if bm.sum() > larea:
                larea = bm.sum()
                lm = x            
    if larea == 0:
        return None
    # get center of x 
    return get_center(seg == lm)

def get_target(traj_path, img_indx, is_annot_validfn, base_pos):
    src_depth = np.load(os.path.join(traj_path, "depth/{:05d}.npy".format(img_indx)))
    src_label = np.load(os.path.join(traj_path, "seg/{:05d}.npy".format(img_indx)))
    
    # pick the larger mask, get centroid for that 
    c = get_center_of_larger_mask(src_label, is_annot_validfn)
    if c is None:
        return None
        
    height, width = src_depth.shape
    uv_one_in_cam, intrinsic_mat, rot, trans = compute_uvone(height, width)
    pts = convert_depth_to_pcd(src_depth, base_pos, uv_one_in_cam, rot, trans)
    pts = pts.reshape((height, width, 3))
    
    xyz_pyrobot = pts[c[0],c[1]]
    return xyz_pyrobot_to_canonical_coords(xyz_pyrobot).tolist()

    
def process(traj_path, out_dir, gt, s, is_annot_validfn):
    src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
    print(f'src_img_ids {src_img_ids}')
    
    reexplore_task_data = {}
    
    base_poses = []
    base_poses_hab = []
    if os.path.isfile(os.path.join(traj_path, 'data_hab.json')):
        with open(os.path.join(traj_path, 'data_hab.json'), 'r') as f:
            data = json.load(f)
            for x in src_img_ids:
                base_poses_hab.append(data[str(x)])
                
    if os.path.isfile(os.path.join(traj_path, 'data.json')):
        with open(os.path.join(traj_path, 'data.json'), 'r') as f:
            data = json.load(f)
            for x in src_img_ids:
                base_poses.append(data[str(x)])
    
    # get target xyz
    target_xyz = []
    for i in range(len(src_img_ids)):
        target_xyz.append(get_target(traj_path, src_img_ids[i], is_annot_validfn, base_poses[i]))
    
    # base_poses = [habitat_base_pos(x) for x in base_poses]
    for i in range(len(src_img_ids)):
        visualize_instances(traj_path, [src_img_ids[i]], is_annot_validfn)
        # print(f'start pose {base_poses[i]} \nspawn {base_poses_hab[i]} \ntarget {target_xyz[i]}\n')
        reexplore_task_data[i] = {
            'src_img_id': src_img_ids[i],
            'spawn_pos': base_poses_hab[i],
            'base_pos': base_poses[i],
            'target': target_xyz[i]
        }
        
    with open(os.path.join(out_dir, 'reexplore_data.json'), 'w') as f:
        json.dump(reexplore_task_data, f)

    with open(os.path.join(out_dir, 'traj_path.txt'), 'w') as f:
        f.write(traj_path)

def find_spawn_loc(baseline_root, outdir):
    jobs = []
    print(f"baseline_root {baseline_root}")
    with executor.batch():
        for traj_path in glob.glob(baseline_root + '/*'):
            if traj_path.split('/')[-1].isdigit():
                print(f'processing {traj_path}')
                traj_id = '/'.join(traj_path.split('/')[-2:])
                for setting in ['class', 'instance']:
                    annot_fn = is_annot_validfn_class if setting == 'class' else is_annot_validfn_inst
                    
                    def job_unit(traj_path, outdir, traj_id, annot_fn, setting):
                        s = SampleGoodCandidates(traj_path, annot_fn, setting)
                        for gt in range(5,30,5):
                            outr = os.path.join(outdir, traj_id, setting, str(gt))
                            os.makedirs(outr, exist_ok=True)
                            print(f'outr {outr}')
                            process(traj_path, outr, gt, s, annot_fn)

                    job = executor.submit(job_unit, traj_path, outdir, traj_id, annot_fn, setting)
                    jobs.append(job)

    print(f"Job Id {jobs[0].job_id.split('_')[0]}, num jobs {len(jobs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_dir",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument("--out_dir", type=str, default="", help="")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.out_dir, 'slurm_logs'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
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

    find_spawn_loc(args.data_dir, args.out_dir)