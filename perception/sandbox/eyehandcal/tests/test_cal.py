#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import json

import torch
torch.set_printoptions(linewidth=160)
import matplotlib.pyplot as plt
import pytest

from eyehandcal.utils import detect_corners, build_proj_matrix, sim_data, mean_loss, \
    quat2rotvec, find_parameter, rotmat, hand_marker_proj_world_camera, uncompress_image

localpath=os.path.abspath(os.path.dirname(__file__))


def test_with_sim_data():
    K = build_proj_matrix(fx=613.9306030273438,  fy=614.3072713216146, ppx=322.1438802083333, ppy=241.59906514485678)

    #add noise
    noise_sigma = 5.0
    obs_data_std, gt_param = sim_data(n=100, K=K, noise_std=noise_sigma)

    L = mean_loss(obs_data_std, gt_param, K)

    param=torch.zeros(9, dtype=torch.float64, requires_grad=True)
    L = lambda param: mean_loss(obs_data_std, param, K)
    print('init param  loss', L(param).item())
    param_star=find_parameter(param, L)

    assert L(param_star) < noise_sigma * 2

    print('found param loss', L(param_star).item(), param_star)
    print('truth param loss', L(gt_param).item(), gt_param)


@pytest.fixture(scope='module')
def collected_data():
    # please download from https://drive.google.com/file/d/1w-2jA6jEMqmhrGqt33ClKc_jGCUuyZnL/view?usp=sharing
    with open(os.path.join(localpath,'caldata_jpeg.pkl'), 'rb') as f:
        data=pickle.load(f)
    uncompress_image(data)
    return data


@pytest.fixture(scope='module')
def data_with_corners(collected_data):
    data_with_corners = detect_corners(collected_data)
    return data_with_corners

def test_plot_corners(data_with_corners):
    corner_count = 0
    for i, d in enumerate(data_with_corners[:3]):
        plt.figure(figsize=(16,10))
        for j, img in enumerate(d['imgs']):
            plt.subplot(1,3,j+1)
            plt.imshow(img[:,:,::-1])
            if d['corners'][j] is not None:
                x,y = d['corners'][j]
                plt.plot(x,y, 'r+')
                plt.title(f'{x:.2f}, {y:.2f}')
                corner_count += 1

    assert corner_count > 0       
    plt.savefig('corner_detection.pdf')
    


def extract_obs_data_std(data, camera_index):
    """
    helper function
    """
    obs_data_std = []
    for d in data:
        if d['corners'][camera_index] is not None:
            obs_data_std.append((
                torch.tensor(d['corners'][camera_index], dtype=torch.float64),
                d['pos'].double(),
                quat2rotvec(d['ori'].double())
            ))

    ic = list(data[0]['intrinsics'].values())[camera_index]
    K=build_proj_matrix(
        fx=ic['fx'],
        fy=ic['fy'],
        ppx=ic['ppx'],
        ppy=ic['ppy'])
    return obs_data_std, K


@pytest.fixture
def params_from_data(data_with_corners):
    num_of_camera=3
    params=[]
    for i in range(num_of_camera):
        print(f'Solve camera {i} pose and ee marker pos')
        obs_data_std, K = extract_obs_data_std(data_with_corners, i)
        print('number of image with marker', len(obs_data_std))
        param=torch.zeros(9, dtype=torch.float64, requires_grad=True)
        L = lambda param: mean_loss(obs_data_std, param, K)
        param_star=find_parameter(param, L)
        print('found param_star loss', L(param_star).item())
        params.append(param_star)

    with torch.no_grad():
        print(torch.stack(params))
    return params

def test_plot_reproj_error(params_from_data, collected_data):
    plt.figure(figsize=(9,3))
    num_of_camera = len(params_from_data)
    for camera_index in range(num_of_camera):
        ax=plt.subplot(1,3, camera_index+1)
        obs_data_std, K = extract_obs_data_std(collected_data, camera_index)
        err=[]
        for obs_marker, pos_ee_base, ori_ee_base in obs_data_std:
            with torch.no_grad():
                proj_marker = hand_marker_proj_world_camera(params_from_data[camera_index], pos_ee_base, ori_ee_base, K)
            
            err.append((proj_marker-obs_marker).norm())
            plt.plot((obs_marker[0], proj_marker[0]),
                    (obs_marker[1], proj_marker[1]),
                    '-'
                    )
        ax.set(xlim=(0, 640), ylim=(480, 0))
        ax.set_aspect('equal','box')
        errs = torch.stack(err)
        ax.set_title(f'cam{camera_index} reproj_err mean:{errs.mean():.2f} max:{errs.max():.2}]')
    plt.savefig('reproj_err.pdf')
    plt.close()



def test_plot_3d_marker(params_from_data, collected_data):
    import sophus as sp
    from fairotag.scene import SceneViz
    def makepose(rotvec, trans):
        return sp.SE3(rotmat(rotvec).detach().numpy(), trans.detach().numpy())

    sv = SceneViz()
    for i, param in enumerate(params_from_data):
        camera_base_ori = param[:3]
        camera_base_pos = param[3:6]
        
        sv.draw_camera(makepose(camera_base_ori, camera_base_pos), .3)
        sv.draw_axes(sp.SE3(), .5)
        
        #plot markers 
        obs_data_std, K = extract_obs_data_std(collected_data, i)
        p_marker_ee = params_from_data[i][6:9]
        for d in obs_data_std:        
            p_marker_base = rotmat(d[2]).matmul(p_marker_ee) + d[1]
            pose_marker_base = makepose(torch.zeros(3), p_marker_base)
            sv.draw_marker(pose_marker_base, 0, 0.01, color=['r','g','b'][i])
            
    sv.show()
    plt.savefig('camera_pose.pdf')


from eyehandcal.scripts import collect_data_and_cal
testcases = [
    [f'--datafile={os.path.join(localpath,"caldata_world_cam.pkl")}', '--marker-id=0'],
    [f'--datafile={os.path.join(localpath,"caldata_wrist_cam.pkl")}', '--marker-id=1', '--proj-func=world_marker_proj_hand_camera'],
]
@pytest.mark.parametrize("argv", testcases, ids = [x[0] for x in testcases])
def test_collect_data_and_cal(argv):
    collect_data_and_cal.main(argv)
    with open('calibration.json', 'rb') as f:
        results = json.load(f)
        for result in results:
            assert result['pixel_error'] < 2.0

non_converging_testcases = [
    [f'--datafile={os.path.join(localpath,"caldata_world_cam.pkl")}', '--marker-id=99'],
    [f'--datafile={os.path.join(localpath,"caldata_world_cam.pkl")}', '--marker-id=0', '--pixel-tolerance=1.0'],
]
@pytest.mark.parametrize("argv", non_converging_testcases, ids = [x[0] for x in non_converging_testcases])
def test_collect_data_and_cal_just_run_for_coverage(argv):
    collect_data_and_cal.main(argv)
