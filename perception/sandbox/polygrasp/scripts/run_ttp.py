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

import time
import os
from unicodedata import category

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
from temporal_task_planner.utils.data_structure_utils import construct_rigid_instance, construct_act_instance
from temporal_task_planner.constants.gen_sess_config.lookup import dishwasher_part_poses, link_id_names, bounding_box_dict, category_vocab, special_instance_vocab
from temporal_task_planner.data_structures.state import State # Instance, RigidInstance
from temporal_task_planner.transform_hardware_to_sim import * # process_hw_pose_for_sim_axes
from temporal_task_planner.utils.datasetpytorch_utils import get_temporal_context
from temporal_task_planner.policy.learned_policy import PromptSituationLearnedPolicy  as PromptSituationPolicy
from temporal_task_planner.trainer.transformer.dual_model import TransformerTaskPlannerDualModel 
from temporal_task_planner.trainer.transformer.configs import TransformerTaskPlannerConfig as PromptSituationConfig
# from temporal_task_planner.trainer.submodule.category_encoder import CategoryEncoderMLP as CategoryEncoder
# from temporal_task_planner.trainer.submodule.pose_encoder import PoseEncoderFourierMLP as PoseEncoder
# from temporal_task_planner.trainer.submodule.temporal_encoder import TemporalEncoderEmbedding as TemporalEncoder
time_to_go = 5

def load_json(pathname):
    with open(pathname, 'r') as f:
        data = json.load(f)
    return data

map_cat_hw_to_sim = {
    'dark_blue_plate': "frl_apartment_plate_01_small_",
    'yellow_cup': "frl_apartment_kitchen_utensil_06_",
    'light_blue_bowl' : "frl_apartment_bowl_03_",
    # 'peach_big_bowl': "frl_apartment_bowl_03_",
    # 'red_bowl': "frl_apartment_bowl_07_small_",  
    'pink_small_bowl': "frl_apartment_bowl_07_small_",  
    "bottom": "ktc_dishwasher_:0000_joint_1",
    "top": "ktc_dishwasher_:0000_joint_3",
}
map_cat_sim_to_hw = {v: k for k, v in map_cat_hw_to_sim.items()}
X = np.load(os.path.expanduser('~') + '/temporal_task_planner/hw_transforms/pick_counter.npy')

artag_lib = {}
artag_lib.update(load_json('data/all_closed.json'))
artag_lib.update(load_json('data/all_open.json'))
artag_lib.update(load_json('data/open_top_drawer.json'))
artag_lib.update(load_json('data/open_bottom_drawer.json'))

top_closed_1 = torch.tensor([-2.8709,  1.7132,  1.3774, -1.8681, -1.4531,  1.9806,  0.4803])
top_closed_2 = torch.tensor([-2.5949,  1.7388,  1.0075, -1.6537, -1.4691,  2.3417,  0.4605])
top_open = torch.tensor([-2.8362,  1.7326,  1.0338, -1.2461, -1.4473,  2.2300, -0.0111])
bottom_closed_1 = torch.tensor([-2.7429,  1.7291,  1.0249, -1.3325, -1.0166,  2.1604, -0.1720])
bottom_closed_2 = torch.tensor([-2.1674,  1.6435,  0.5143, -1.2161, -0.9969,  2.5138,  0.2471])
#torch.tensor([-2.2719,  1.675,  0.6623, -1.1511, -0.6550,  2.4067, -0.1707])
bottom_open = torch.tensor([-2.4781,  1.7273,  0.5201, -0.9822, -0.9824,  2.4964, -0.0416])
# torch.tensor([-2.2077,  1.65,  0.2767, -0.7902, -0.6421,  2.5545, -0.2362])
high_position_close = torch.tensor([-2.8740,  1.3173,  1.5164, -1.2091, -1.1478,  1.4974, -0.1642])
sink_pose = torch.tensor([-0.2135, -0.0278,  0.5381, -2.1573,  0.0384,  2.1235, -0.6401])
# torch.tensor([-1.1165,  0.7988,  1.5438, -2.3060, -1.0097,  2.0797, -0.5347])

act_instance = construct_act_instance(is_action_available=False, is_action_to_be_predicted=True, relative_timestep=1)

def move_to_joint_pos(robot, pos, time_to_go=5.0):
    state_log = []
    while len(state_log) < time_to_go*100:
        state_log = robot.move_to_joint_positions(pos, time_to_go)
    return state_log
    

def open_bottom_drawer(robot):
    traj = robot.move_to_joint_positions(high_position_close)
    traj = robot.move_to_joint_positions(bottom_closed_1)
    traj = robot.move_to_joint_positions(bottom_closed_2)
    traj = robot.move_to_joint_positions(bottom_open)
    traj = robot.move_to_joint_positions(high_position_close)

def open_top_drawer(robot):
    traj = robot.move_to_joint_positions(high_position_close)
    traj = robot.move_to_joint_positions(top_closed_1)
    traj = robot.move_to_joint_positions(top_closed_2)
    traj = robot.move_to_joint_positions(top_open)
    traj = robot.move_to_joint_positions(high_position_close)
    
def close_top_drawer(robot):
    traj = robot.move_to_joint_positions(high_position_close)
    traj = robot.move_to_joint_positions(top_open)
    traj = robot.move_to_joint_positions(top_closed_2)
    traj = robot.move_to_joint_positions(top_closed_1)
    traj = robot.move_to_joint_positions(high_position_close)

def close_bottom_drawer(robot):
    traj = robot.move_to_joint_positions(high_position_close)
    traj = robot.move_to_joint_positions(bottom_open)
    traj = robot.move_to_joint_positions(bottom_closed_2)
    traj = robot.move_to_joint_positions(bottom_closed_1)
    traj = robot.move_to_joint_positions(high_position_close)

def view_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])
    return
  
def save_depth_images(rgbd):
    depth_img = rgbd[:,:,-1]
    
    
  
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


def  is_shadow(val, epsilon=5e-3):
    err_01 = abs(val[0] - val[1]) 
    err_21 = abs(val[2] - val[1])
    err_02 = abs(val[2] - val[0])
    if err_01 < epsilon and err_21 < epsilon and err_02 < epsilon:
        return True
    return False
         
def is_not_on_counter(xyz):
    if xyz[-1] > 0:
        return False
    return True

def get_category_to_pcd_map(obj_pcds, cur_data, ref_data):
    category_to_pcd_map = {cat: [] for cat in ref_data}
    for idx, val in cur_data.items():
        min_err = 10000
        min_err_category = 'ignore'  # default
        if is_shadow(val[0]) or is_not_on_counter(obj_pcds[idx].get_center()):
            continue
        for category, rgb_value in ref_data.items():
            err = np.linalg.norm(np.array(val) - np.array(rgb_value))
            if min_err > err:
                min_err = err 
                min_err_category = category 
        if min_err_category.startswith('ignore_'):
            continue
        category_to_pcd_map[min_err_category].append(obj_pcds[idx])
    return category_to_pcd_map


def save_rgbd(rgbd):
    num_cams = rgbd.shape[0]
    f, axarr = plt.subplots(2, num_cams)

    for i in range(num_cams):
        if num_cams > 1:
            ax1, ax2 = axarr[0, i], axarr[1, i]
        else:
            ax1, ax2 = axarr
        ax1.imshow(rgbd[i, :, :, :3].astype(np.uint8))
        ax2.matshow(rgbd[i, :, :, 3:])

    f.savefig("rgbd.png")
    plt.close(f)
    
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


def execute_grasp(robot, chosen_grasp, time_to_go, place_in_drawer=False):
    traj, success = robot.grasp(
        chosen_grasp, time_to_go=time_to_go, gripper_width_success_threshold=0.001
    )
    print(f"Grasp success: {success}")
    breakpoint()
    if success:
        ## place in sink
        curr_pose, curr_ori = robot.get_ee_pose()
        print("Moving up")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0, 0.3]),
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        curr_pose, curr_ori = robot.get_ee_pose()

        if place_in_drawer:
            print("Placing object in hand in drawer...")
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
        else:
            print("Moving to sink pose")
            traj += move_to_joint_pos(robot, sink_pose)
            
    print("Opening gripper")
    robot.gripper_open()
    curr_pose, curr_ori = robot.get_ee_pose()
    print("Moving up")
    traj += robot.move_until_success(
        position=curr_pose + torch.Tensor([0, 0.0, 0.3]),
        orientation=curr_ori,
        time_to_go=time_to_go,
    )
    robot.go_home()
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

        # # # Define some parameters for each workspace.
        # if cam_i == 0:
        #     masks = masks_1
        #     hori_offset = torch.Tensor([0, -0.4, 0])
        # else:
        #     masks = masks_2
        #     hori_offset = torch.Tensor([0, 0.4, 0])
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
            
            
            save_rgbd(rgbd)

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
            # 
            
            
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
            ref_data = load_json(pathname=Path(
                hydra.utils.get_original_cwd(),
                'data', 
                'category_to_rgb_map.json'
            ).as_posix())
                                            
            # 
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

            traj = execute_grasp(robot, chosen_grasp, time_to_go)

            print("Going home")
            robot.go_home()

def transform_hw_to_sim_pose(pcd):
    """converts the 3d coordinates in hw robot frame to 
    sim's default world frame
    1. load the transform
    2. convert hw coordinates in correct format for multiply 
    """
    # xyz_hw_in_sim_axes
    xyz_A = process_hw_pose_for_sim_axes(pcd.get_center())
    b = get_simulation_coordinates(xyz_A=xyz_A, X=X)
    return b + [1.0,0.0,0.0,0.0] # default quaternion
    
# def transform_hw_to_sim_category(category_to_pcd_map):

#     category_name=[map_cat_hw_to_sim[key] for key in category_to_pcd_map]
#     category_bb = [map_cat_tok_to_bb[cat] for cat in category_name]
#     return {
#         'category_name': category_name,
#         'category': category_bb
#     }
    
def get_hardware_countertop_objects(category_to_pcd_map):
    rigid_instances = []
    for colorbasedname, pcd_list in category_to_pcd_map.items():
        for i, pcd in enumerate(pcd_list):
            category_name = map_cat_hw_to_sim[colorbasedname]
            name = "{}:{:04d}".format(category_name, i)
            pose = transform_hw_to_sim_pose(pcd)
            entry = construct_rigid_instance(name, pose, category_name, relative_timestep=1) 
            # RigidInstance(
            #     timestep=1,
            #     category=bounding_box_dict[category_name],
            #     pose=pose,
            #     action_masks=False,
            #     is_real=True,
            #     category_token_name=category_name,  # record utensil type
            #     category_token=category_vocab.word2index(category_name),
            #     instance_token=special_instance_vocab.word2index(name),
            #     category_name=category_name,
            #     instance_name=name,
            # )
            rigid_instances.append(entry)
    return rigid_instances
    
        
def get_category_pcd_on_counter(cfg, cameras, segmentation_client):
    rgbd = cameras.get_rgbd()
    scene_pcd = cameras.get_pcd(rgbd)
    print("Segmenting image...")
    unmerged_obj_pcds = []
    for i in range(cameras.n_cams):
        obj_masked_rgbds, obj_masks = segmentation_client.segment_img(
            rgbd[i], min_mask_size=cfg.min_mask_size
        )
        unmerged_obj_pcds += [
            cameras.get_pcd_i(obj_masked_rgbd, i)
            for obj_masked_rgbd in obj_masked_rgbds
        ]
    obj_pcds = merge_pcds(unmerged_obj_pcds)
    cur_data = get_object_detect_dict(obj_pcds)
    ref_data = load_json(pathname=Path(
        hydra.utils.get_original_cwd(),
        'data', 
        'category_to_rgb_map.json'
    ).as_posix())                                    
    # 
    category_to_pcd_map = get_category_to_pcd_map(obj_pcds, cur_data, ref_data)
    # rigid_instances = get_hardware_rigid_instance(category_to_pcd_map)
    # # # instance_name_list = # count instance name
    # # category_dict = transform_hw_to_sim_category(category_to_pcd_map)
    # # pose_list = transform_hw_to_sim_pose(category_to_pcd_map)
    # # timestep_list = [1.]*len(pose_list)
    # # is_real_list = [1.]*len(pose_list)
    # # State()
    # 
    
    # situation = {
    #     'timestep': torch.tensor([timestep_list])
    # }
    # # transform_position_counter_hw2sim = np.load(os.path.expanduser('~') + '/temporal_task_planner/hw_transforms/pick_counter.npy')
    # # transform_category_hw2sim = 
    # # # return category_to_pcd_map
    return category_to_pcd_map

def get_marker_corners(cameras, frt_cams):
    rgbd = cameras.get_rgbd()
    rgbd_masked = rgbd
    save_rgbd(rgbd)
    uint_rgbs = rgbd[:,:,:,:3].astype(np.uint8)
    drawer_markers = frt_cams[drawer_camera_index].detect_markers(uint_rgbs[drawer_camera_index])
    # print(drawer_markers)
    data = [{
            "id": int(m.id), 
            "corner": m.corner.astype(np.int32).tolist()  
        } for m in drawer_markers]
    return data

def get_drawer_as_dishwasher_instances(cameras, frt_cams, drawer_status):
    # measure similarity to the 4 scenarios of artags 
    # and set the drawer poses in sim
    data = get_marker_corners(cameras, frt_cams)
    # # check the position of marker tags corners and take the most similar
    # most_likely_status = 'all_closed' 
    # min_err = 10000000
    # for key, markers in artag_lib.items():
    #     for ref_marker_info in markers:
    #         for cur_marker_info in data:
    #             if cur_marker_info['id'] == ref_marker_info['id']:
    #                 # compare the norm of the difference in corner matrices
    #                 flag_match = True
    #             else:
    #                 flag_match = False
    #     if flag_match:
    #         err = np.linalg.norm(cur_marker_info['corner'] - ref_marker_info['corner'])
    #         if min_err > err:
    #             min_err = err
    #             most_likely_status = key
    door = construct_rigid_instance(
        name=link_id_names["door"],
        pose=dishwasher_part_poses['door']['open'],
        category_name=link_id_names["door"],
        relative_timestep=1
    )
                                
    # if status == 'all_closed':
    # default
    bottom_rack = construct_rigid_instance(
        name=link_id_names['bottom'],
        pose=dishwasher_part_poses['bottom']['close'],
        category_name=link_id_names['bottom'],
        relative_timestep=1
    )
    top_rack = construct_rigid_instance(
        name=link_id_names["top"],
        pose=dishwasher_part_poses['top']['close'],
        category_name=link_id_names["top"],
        relative_timestep=1
    )
    if drawer_status['top'] == 'open':
        top_rack = construct_rigid_instance(
            name=link_id_names["top"],
            pose=dishwasher_part_poses['top']['open'],
            category_name=link_id_names["top"],
            relative_timestep=1
        )
    elif drawer_status['bottom'] == 'open':
        bottom_rack = construct_rigid_instance(
            name=link_id_names['bottom'],
            pose=dishwasher_part_poses['bottom']['open'],
            category_name=link_id_names['bottom'],
            relative_timestep=1
        )
    return [door, bottom_rack, top_rack]

@hydra.main(config_path="../conf", config_name="run_ttp")
def main(cfg):
    drawer_status = {
        'top': 'close',
        'bottom': 'close'
    }
    print('drawer_status: ', drawer_status)
    print('Do you want to modify drawer_status ???')
    breakpoint()
    # config = hydra.utils.instantiate(cfg.config)
    config = PromptSituationConfig(
        num_instances=60,
        d_model=cfg.d_model,
        nhead=2,
        d_hid=cfg.d_hid,
        num_slots=cfg.num_slots,
        slot_iters=cfg.slot_iters,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dropout=0.,
        batch_first=True,
        category_embed_size=cfg.category_embed_size,
        pose_embed_size=cfg.pose_embed_size,
        temporal_embed_size=cfg.temporal_embed_size,
        marker_embed_size=cfg.marker_embed_size,
    )
    category_encoder = hydra.utils.instantiate(cfg.category_encoder)
    temporal_encoder = hydra.utils.instantiate(cfg.temporal_encoder)
    pose_encoder = hydra.utils.instantiate(cfg.pose_encoder)
    reality_marker_encoder = hydra.utils.instantiate(cfg.reality_marker_encoder)
    model = TransformerTaskPlannerDualModel(
        config, 
        category_encoder=category_encoder,
        temporal_encoder=temporal_encoder,
        pose_encoder=pose_encoder,
        reality_marker_encoder=reality_marker_encoder,
    )
    model.load_state_dict(torch.load(cfg.checkpoint_path)['model_state_dict']) #, map_location='cpu')
    policy = PromptSituationPolicy(model, pick_only=False, device='cpu')
    
    prompt_temporal_context = get_temporal_context(cfg.prompt_session_path)
    prompt_temporal_context.pick_only = True
    prompt = prompt_temporal_context.process_states()
    for key, val in prompt.items():
        prompt[key] = torch.tensor(val).unsqueeze(0)
    prompt_input_len = len(prompt["timestep"][0])
    prompt["src_key_padding_mask"] = (
        torch.zeros(prompt_input_len).bool().unsqueeze(0)
    )
    policy.reset(prompt)
    
    print(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    print(f"Current working directory: {os.getcwd()}")

    print("Initialize robot & gripper")
    robot = hydra.utils.instantiate(cfg.robot)
    robot.gripper_open()
    robot.go_home()

    print("Initializing cameras")
    cfg.cam.intrinsics_file = hydra.utils.to_absolute_path(cfg.cam.intrinsics_file)
    cfg.cam.extrinsics_file = hydra.utils.to_absolute_path(cfg.cam.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.cam)

    
    print("Loading camera workspace masks")
    # # import pdb; pdb.set_trace()
    masks_1 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_1],
        dtype=np.float64,
    )
    masks_1[1,:,:] *=0
    masks_2 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_2],
        dtype=np.float64,
    )
    # masks_1[:2,:,:] *=0
    masks_1[0,:,:] *=0
    masks_1[2,:,:] *=0

    # if cam_i == 0:
    #     masks = masks_1
    #     hori_offset = torch.Tensor([0, -0.4, 0])
    # else:
    #     masks = masks_2
    #     hori_offset = torch.Tensor([0, 0.4, 0])
    hori_offset = torch.Tensor([0.,0.,0.])

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
    MARKER_ID = [22, 27, 26, 23, 29] #[18]
    for i in MARKER_ID:
        for frt in frt_cams:
            frt.register_marker_size(i, MARKER_LENGTH)
    step = -1

    while True:
        step += 1    
        rgbd = cameras.get_rgbd()
        scene_pcd = cameras.get_pcd(rgbd)
        
        category_to_pcd_map = get_category_pcd_on_counter(cfg, cameras, segmentation_client)
        # save categories seen with their point clouds.
        objdetect_on_counter = []
        for cat, pcd_list in category_to_pcd_map.items():
            if len(pcd_list):
                objdetect_on_counter.append(cat)
                for i, pcd in enumerate(pcd_list):
                    o3d.io.write_point_cloud(f'step-{step}_{cat}-{i}.pcd', pcd) 
                
        dw_parts = get_drawer_as_dishwasher_instances(cameras, frt_cams, drawer_status)
        dishes = get_hardware_countertop_objects(category_to_pcd_map)
        rigid_instances = dw_parts + dishes

        current_state = State(rigid_instances=rigid_instances, act_instances=act_instance)
        print(current_state)
        action = policy.get_action(current_state)
        breakpoint()
        print(map_cat_sim_to_hw.get(action.pick_instance.category_name, None)) 
        if action.pick_instance.category_name == link_id_names["door"]:  # "ktc_dishwasher_:0000_joint_2":
            break
        elif action.pick_instance.category_name == link_id_names["bottom"]: #  "ktc_dishwasher_:0000_joint_1":
            if abs(action.get_place_position()[2] - 0.95) < 1e-3:
                if drawer_status['top'] == 'open':
                    break
                open_bottom_drawer(robot)
                drawer_status['bottom'] = "open"
            else:
                if drawer_status['top'] == 'open':
                    break
                close_bottom_drawer(robot)
                drawer_status['bottom'] = "close"
        elif action.pick_instance.category_name == link_id_names["top"]:  # "ktc_dishwasher_:0000_joint_3":
            if abs(action.get_place_position()[2] - 0.95) < 1e-3:
                open_top_drawer(robot)
                drawer_status['top'] = "open"
            else:
                close_top_drawer(robot)
                drawer_status['top'] = "close"
        elif action.pick_instance.category_name in map_cat_sim_to_hw:
            _cat = map_cat_sim_to_hw[ action.pick_instance.category_name]
            _pcd = category_to_pcd_map[_cat][0]
            if _pcd is None:
                # break #point()
                return 2
            # replace obj_pcds with id_to_pcd[id] for grasping selected id
            # for _cat, _pcd in category_to_pcd_map.items():
            # print(f'Grasping {_cat}')
            
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
            if abs(action.get_place_position()[1] - 0.9599) < 1e-3:
                place_in_drawer = False 
            else:
                place_in_drawer = True 
            traj = execute_grasp(robot, chosen_grasp, time_to_go, place_in_drawer)

            print("Going home")
            robot.go_home()
  
        


if __name__ == "__main__":
    main()
