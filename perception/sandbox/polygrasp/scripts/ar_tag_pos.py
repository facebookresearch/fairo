#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""


#old
# high_position_open= torch.tensor([-2.8628,  1.7270,  1.8333, -1.6115, -1.6032,  1.4454,  0.4743])
# top closed1 :tensor([-2.6614,  1.7002,  1.3380, -1.7611, -1.6574,  1.8371,  0.7491])
# top closed2 : tensor([-2.4981,  1.7320,  1.0551, -1.6264, -1.6605,  2.1295,  0.7341])
# bottom closed1: tensor([-2.5299,  1.6819,  0.8728, -1.4600, -1.5670,  2.3802, -2.7360])
# bottom closed2: tensor([-2.3284,  1.7304,  0.6423, -1.2444, -1.4370,  2.5559, -2.7209])
# top open: tensor([-2.7492,  1.7202,  1.0310, -1.1538, -1.3008,  2.1873, -2.8715])
# bottom open: tensor([-2.4589,  1.7582,  0.5844, -0.9734, -1.1897,  2.4049, -2.8648])
drawer_camera_index = 1
top_rack_ar_tag = [26, 22, 27]
bottom_rack_ar_tag = [23,29]

import time
import os

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

import json
import open3d as o3d
from pathlib import Path
from scipy.stats import mode
from functools import partial 

import polygrasp
from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img

import fairotag

# top_rack_open_ar_tag = [-0.05914402, -0.60028556, -0.38374834]

top_closed_1 = torch.tensor([-2.8709,  1.7132,  1.3774, -1.8681, -1.4531,  1.9806,  0.4803])
top_closed_2 = torch.tensor([-2.5949,  1.7388,  1.0075, -1.6537, -1.4691,  2.3417,  0.4605])
top_open = torch.tensor([-2.8362,  1.7326,  1.0338, -1.2461, -1.4473,  2.2300, -0.0111])
bottom_closed_1 = torch.tensor([-2.7429,  1.7291,  1.0249, -1.3325, -1.0166,  2.1604, -0.1720])
bottom_closed_2 = torch.tensor([-2.2719,  1.675,  0.6623, -1.1511, -0.6550,  2.4067, -0.1707])
bottom_open = torch.tensor([-2.2077,  1.65,  0.2767, -0.7902, -0.6421,  2.5545, -0.2362])
high_position_close = torch.tensor([-2.8740,  1.3173,  1.5164, -1.2091, -1.1478,  1.4974, -0.1642])

def move_to_joint_pos(robot, pos, time_to_go=5.0):
    state_log = []
    while len(state_log) < time_to_go*100:
        state_log = robot.move_to_joint_positions(pos, time_to_go)
    return state_log
    

def open_bottom_drawer(robot):
    traj = move_to_joint_pos(robot, high_position_close)
    traj = move_to_joint_pos(robot, bottom_closed_1)
    traj = move_to_joint_pos(robot, bottom_closed_2)
    traj = move_to_joint_pos(robot, bottom_open)
    traj = move_to_joint_pos(robot, high_position_close)

def open_top_drawer(robot):
    traj = move_to_joint_pos(robot, high_position_close)
    traj = move_to_joint_pos(robot, top_closed_1)
    traj = move_to_joint_pos(robot, top_closed_2)
    traj = move_to_joint_pos(robot, top_open)
    traj = move_to_joint_pos(robot, high_position_close)
    
def close_top_drawer(robot):
    traj = move_to_joint_pos(robot, high_position_close)
    traj = move_to_joint_pos(robot, top_open)
    traj = move_to_joint_pos(robot, top_closed_2)
    traj = move_to_joint_pos(robot, top_closed_1)
    traj = move_to_joint_pos(robot, high_position_close)

def close_bottom_drawer(robot):
    traj = move_to_joint_pos(robot, high_position_close)
    traj = move_to_joint_pos(robot, bottom_open)
    traj = move_to_joint_pos(robot, bottom_closed_2)
    traj = move_to_joint_pos(robot, bottom_closed_1)
    traj = move_to_joint_pos(robot, high_position_close)

def view_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])
    return
  
  
def get_object_detect_dict(obj_pcds):
    data = {}
    for idx, pcd in enumerate(obj_pcds):
        data[idx] = mode(pcd.colors, axis=0).mode.tolist()
    return data 
    
    
# only to record the rgb mode values and manually label
def save_object_detect_dict(obj_pcds):
    data = get_object_detect_dict(obj_pcds)
    with open('category_to_rgb_mapping.json', 'w') as f:
        json.dump(data, f, indent=4)
    return 


def read_object_detect_dict(pathname):
    with open(pathname, 'r') as f:
        data = json.load(f)
    return data

def  is_shadow(val, epsilon=5e-2):
    err_01 = abs(val[0] - val[1]) 
    err_21 = abs(val[2] - val[1])
    err_02 = abs(val[2] - val[0])
    if err_01 < epsilon and err_21 < epsilon and err_02 < epsilon:
        return True
    return False
         

def get_category_to_pcd_map(obj_pcds, cur_data, ref_data):
    category_to_pcd_map = {}
    for idx, val in cur_data.items():
        min_err = 1000
        min_err_category = 'ignore'  # default
        if is_shadow(val[0]):
            continue
        for category, rgb_value in ref_data.items():
            err = np.linalg.norm(np.array(val) - np.array(rgb_value))
            if min_err > err:
                min_err = err 
                min_err_category = category 
        category_to_pcd_map[min_err_category] = obj_pcds[idx]
    return category_to_pcd_map


def save_rgbd_masked(rgbd, rgbd_masked):
    num_cams = rgbd.shape[0]
    f, axarr = plt.subplots(2, num_cams)

    for i in range(num_cams):
        if num_cams > 1:
            ax1, ax2 = axarr[0, i], axarr[1, i]
        else:
            ax1, ax2 = axarr
        ax1.imshow(rgbd[i, :, :, :3].astype(np.uint8))
        ax2.imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))

    f.savefig("rgbd_masked.png")
    plt.close(f)


def get_obj_grasps(grasp_client, obj_pcds, scene_pcd):
    for obj_i, obj_pcd in enumerate(obj_pcds):
        print(f"Getting obj {obj_i} grasp...")
        grasp_group = grasp_client.get_grasps(obj_pcd)
        filtered_grasp_group = grasp_client.get_collision(grasp_group, scene_pcd)
        if len(filtered_grasp_group) < len(grasp_group):
            print(
                "Filtered"
                f" {len(grasp_group) - len(filtered_grasp_group)}/{len(grasp_group)} grasps"
                " due to collision."
            )
        if len(filtered_grasp_group) > 0:
            return obj_i, filtered_grasp_group
    raise Exception(
        "Unable to find any grasps after filtering, for any of the"
        f" {len(obj_pcds)} objects"
    )


def merge_pcds(pcds, eps=0.1, min_samples=2):
    """Cluster object pointclouds from different cameras based on centroid using DBSCAN; merge when within eps"""
    xys = np.array([pcd.get_center()[:2] for pcd in pcds])
    cluster_labels = (
        sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xys).labels_
    )

    # Logging
    total_n_objs = len(xys)
    total_clusters = cluster_labels.max() + 1
    unclustered_objs = (cluster_labels < 0).sum()
    print(
        f"Clustering objects from all cameras: {total_clusters} clusters, plus"
        f" {unclustered_objs} non-clustered objects; went from {total_n_objs} to"
        f" {total_clusters + unclustered_objs} objects"
    )

    # Cluster label == -1 when unclustered, otherwise cluster label >=0
    final_pcds = []
    cluster_to_pcd = dict()
    for cluster_label, pcd in zip(cluster_labels, pcds):
        if cluster_label >= 0:
            if cluster_label not in cluster_to_pcd:
                cluster_to_pcd[cluster_label] = pcd
            else:
                cluster_to_pcd[cluster_label] += pcd
        else:
            final_pcds.append(pcd)

    return list(cluster_to_pcd.values()) + final_pcds


def execute_grasp(robot, chosen_grasp, hori_offset, time_to_go):
    traj, success = robot.grasp(
        chosen_grasp, time_to_go=time_to_go, gripper_width_success_threshold=0.001
    )
    print(f"Grasp success: {success}")
    breakpoint()
    if success:
        print("Placing object in hand to desired pose...")
        curr_pose, curr_ori = robot.get_ee_pose()
        print("Moving up")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0, 0.3]),
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        print("Moving horizontally")
        traj += robot.move_until_success(
            position=torch.Tensor([-0.09, -0.61, 0.2]),
            orientation=[1,0,0,0],
            time_to_go=time_to_go,
        )
        curr_pose, curr_ori = robot.get_ee_pose()
        print("Moving down")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, -0.20]),
            orientation=[1,0,0,0],
            time_to_go=time_to_go,
        )

    print("Opening gripper")
    robot.gripper_open()
    curr_pose, curr_ori = robot.get_ee_pose()
    print("Moving up")
    traj += robot.move_until_success(
        position=curr_pose + torch.Tensor([0, 0.0, 0.3]),
        orientation=curr_ori,
        time_to_go=time_to_go,
    )

    return traj

def pickplace(
    robot,
    category_order,
    cfg,
    masks_1,
    masks_2,
    root_working_dir,
    cameras,
    frt_cams,
    segmentation_client,
    grasp_client,
):
    for outer_i in range(cfg.num_bin_shifts):
        cam_i = outer_i % 2
        print(f"=== Starting bin shift with cam {cam_i} ===")

        # Define some parameters for each workspace.
        if cam_i == 0:
            masks = masks_1
            hori_offset = torch.Tensor([0, -0.4, 0])
        else:
            masks = masks_2
            hori_offset = torch.Tensor([0, 0.4, 0])
        time_to_go = 5

        for i in range(cfg.num_grasps_per_bin_shift):
            # Create directory for current grasp iteration
            os.chdir(root_working_dir)
            timestamp = int(time.time())
            os.makedirs(f"{timestamp}")
            os.chdir(f"{timestamp}")

            print(
                f"=== Grasp {i + 1}/{cfg.num_grasps_per_bin_shift}, logging to"
                f" {os.getcwd()} ==="
            )

            print("Getting rgbd and pcds..")
            rgbd = cameras.get_rgbd()
            rgbd_masked = rgbd

            print("Detecting markers & their corresponding points")
            uint_rgbs = rgbd[:,:,:,:3].astype(np.uint8)
            id_to_pose = {}
            for i, (frt, uint_rgb) in enumerate(zip(frt_cams, uint_rgbs)):
                markers = frt.detect_markers(uint_rgb)
                for marker in markers:
                    w_i, h_i = marker.corner.mean(axis=0).round().astype(int)

                    single_point = np.zeros_like(rgbd[i])
                    single_point[h_i, w_i, :] = rgbd[i, h_i, w_i, :]
                    pcd = cameras.get_pcd_i(single_point, i)
                    xyz = pcd.get_center()
                    id_to_pose[marker.id] = xyz

            scene_pcd = cameras.get_pcd(rgbd)
            save_rgbd_masked(rgbd, rgbd_masked)

            print("Segmenting image...")
            unmerged_obj_pcds = []
            for i in range(cameras.n_cams):
                obj_masked_rgbds, obj_masks = segmentation_client.segment_img(
                    rgbd_masked[i], min_mask_size=cfg.min_mask_size
                )
                unmerged_obj_pcds += [
                    cameras.get_pcd_i(obj_masked_rgbd, i)
                    for obj_masked_rgbd in obj_masked_rgbds
                ]
                
            obj_pcds = merge_pcds(unmerged_obj_pcds)
            # breakpoint()
            
            
            if len(obj_pcds) == 0:
                print(
                    f"Failed to find any objects with mask size > {cfg.min_mask_size}!"
                )
                return 1
            
            # print(f"Finding ARTag")
            # id_to_pcd = {}
            # obj_centers = np.array([x.get_center() for x in obj_pcds])
            # for id, pose in id_to_pose.items():
            #     dists = np.linalg.norm(obj_centers - pose, axis=1)
            #     argmin = dists.argmin()
            #     min = dists[argmin]
            #     if min < 0.05:
            #         id_to_pcd[id] = obj_pcds[argmin]
            # print(f"Found object pointclouds corresponding to ARTag ids {list(id_to_pcd.keys())}")
            cur_data = get_object_detect_dict(obj_pcds)
            ref_data = read_object_detect_dict(pathname=Path(
                hydra.utils.get_original_cwd(),
                'data', 
                'category_to_rgb_map.json'
            ).as_posix())
                                            
            # breakpoint()
            category_to_pcd_map = get_category_to_pcd_map(obj_pcds, cur_data, ref_data)
            for _cat in category_order:
                if _cat not in category_to_pcd_map.keys():
                    continue
                else:
                    break
            # _cat = category_order[idx]
            _pcd = category_to_pcd_map.get(_cat, None)
            if _pcd is None:
                # break #point()
                return 2
            # replace obj_pcds with id_to_pcd[id] for grasping selected id
            # for _cat, _pcd in category_to_pcd_map.items():
            print(f'Grasping {_cat}')
            
            # print("Getting grasps per object...")
            obj_i, filtered_grasp_group = get_obj_grasps(
                grasp_client, [_pcd], scene_pcd
            )

            # print("Choosing a grasp for the object")
            chosen_grasp, final_filtered_grasps = robot.select_grasp(
                filtered_grasp_group
            )
            grasp_client.visualize_grasp(scene_pcd, final_filtered_grasps)
            grasp_client.visualize_grasp(
                _pcd, final_filtered_grasps, name=_cat
            )

            traj = execute_grasp(robot, chosen_grasp, hori_offset, time_to_go)

            print("Going home")
            robot.go_home()
            
def get_marker_corners(cameras, frt_cams, root_working_dir, name):
    rgbd = cameras.get_rgbd()
    rgbd_masked = rgbd
    save_rgbd_masked(rgbd, rgbd_masked)
    uint_rgbs = rgbd[:,:,:,:3].astype(np.uint8)
    drawer_markers = frt_cams[drawer_camera_index].detect_markers(uint_rgbs[drawer_camera_index])
    print(drawer_markers)
    data = {
        name : [{
            "id": int(m.id), 
            "corner": m.corner.astype(np.int32).tolist()  
        } for m in drawer_markers]
    }
    with open(Path(root_working_dir, 'data', name + '.json').as_posix(), 'w') as f:
        json.dump(data, f, indent=2)
    return 

@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    root_working_dir = hydra.utils.get_original_cwd()  # os.getcwd()
    print(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    print(f"Current working directory: {os.getcwd()}")

    print("Initialize robot & gripper")
    robot = hydra.utils.instantiate(cfg.robot)
    robot.gripper_open()
    robot.go_home()
    # robot.move_until_success(position=torch.tensor([ 0.70, -0.07,  0.0101]), orientation=robot.get_ee_pose()[1], time_to_go=10)

    print("Initializing cameras")
    cfg.cam.intrinsics_file = hydra.utils.to_absolute_path(cfg.cam.intrinsics_file)
    cfg.cam.extrinsics_file = hydra.utils.to_absolute_path(cfg.cam.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.cam)

    # print("Loading camera workspace masks")
    # # import pdb; pdb.set_trace()
    # masks_1 = np.array(
    #     [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_1],
    #     dtype=np.float64,
    # )
    # masks_1[-1,:,:] *=0
    # masks_2 = np.array(
    #     [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_2],
    #     dtype=np.float64,
    # )
    # masks_1[:2,:,:] *=0

    print("Connect to grasp candidate selection and pointcloud processor")
    segmentation_client = SegmentationClient()
    grasp_client = GraspClient(
        view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    )

    print("Loading ARTag modules")
    frt_cams = [fairotag.CameraModule() for _ in range(cameras.n_cams)]
    for frt, intrinsics in zip(frt_cams, cameras.intrinsics):
        frt.set_intrinsics(intrinsics)
    MARKER_LENGTH = 0.04
    MARKER_ID = [27, 26, 23, 29]
    for i in MARKER_ID:
        for frt in frt_cams:
            frt.register_marker_size(i, MARKER_LENGTH)
    
    get_marker_corners(cameras, frt_cams, root_working_dir, name='all_closed')
    breakpoint()

    open_bottom_drawer(robot)
    robot.go_home()
    breakpoint()
    
    get_marker_corners(cameras, frt_cams, root_working_dir, name='open_bottom_drawer')
    
    open_top_drawer(robot)
    robot.go_home()
    breakpoint()    
    get_marker_corners(cameras, frt_cams, root_working_dir, name='all_open')

    close_top_drawer(robot)
    close_bottom_drawer(robot)
    open_top_drawer(robot)
    robot.go_home()
    breakpoint()
    get_marker_corners(cameras, frt_cams, root_working_dir, name='open_top_drawer')
    
    close_bottom_drawer(robot)
    robot.go_home()
    breakpoint()
    get_marker_corners(cameras, frt_cams, root_working_dir, name='all_closed')
if __name__ == "__main__":
    main()
