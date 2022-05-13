# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2
import math

def uncompress_image(data):
    for d in data:
        # decode imgs_jpeg_encoded -> imgs
        if 'imgs_jpeg_encoded' in d:
            assert 'imgs' not in d
            d['imgs'] = []
            for img_jpeg_encoded in d['imgs_jpeg_encoded']:
                d['imgs'].append(cv2.imdecode(img_jpeg_encoded, cv2.IMREAD_COLOR))


def detect_corners(data, target_idx=9):
    """
        data: [{'img': [np.ndarray]}]
        return: [{'corners', [(x,y)]}]
    """
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_param = cv2.aruco.DetectorParameters_create()
    aruco_param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    for i,d in enumerate(data):
        d['corners'] = []
        for j, img in enumerate(d['imgs']):
            result=cv2.aruco.detectMarkers(img.astype(np.uint8), dictionary=aruco_dict, parameters=aruco_param)
            corners, idx, rej = result
            if idx is not None and target_idx in idx:
                corner_i = idx.squeeze(axis=1).tolist().index(target_idx)
                target_corner=corners[corner_i][0,0,:].tolist()
                d['corners'].append(target_corner)
            else:
                d['corners'].append(None)
    return data


def skewsym(v):
    """
    pytorch backwark() compatible
    """
    zero=torch.tensor([0.])
    return torch.stack([
     zero, -v[2:3],  v[1:2],
     v[2:3],  zero, -v[0:1],
    -v[1:2],  v[0:1],  zero
    ]).reshape(3,3)


def quat2rotvec(v):
    u = v[:3]
    u = u / u.norm()
    theta = 2 * torch.acos(v[3])
    return u * theta


def rotmat(v):
    assert len(v)==3
    v_ss = skewsym(v)
    return torch.matrix_exp(v_ss)



# TODO: use fairotag.camera.Camera._intrinsic
def build_proj_matrix(fx, fy, ppx, ppy, coeff=None):
    # consider handle distortion here
    return torch.DoubleTensor([[fx, 0., ppx],
                                [0., fy, ppy],
                                [0., 0.,  1.]])


def hand_marker_proj_world_camera(param, pos_ee_base, ori_ee_base, K):
    camera_base_ori = param[:3]
    camera_base_pos = param[3:6]
    p_marker_ee = param[6:9]
    p_marker_camera = rotmat(-camera_base_ori).matmul(
            (rotmat(ori_ee_base).matmul(p_marker_ee) + pos_ee_base)-camera_base_pos)
    p_marker_image = K.matmul(p_marker_camera)
    return p_marker_image[:2]/p_marker_image[2]

def world_marker_proj_hand_camera(param, pos_ee_base, ori_ee_base, K):
    ori_camera_ee = param[:3]
    pos_camera_ee = param[3:6]
    pos_marker_base = param[6:9]
    pos_marker_camera = rotmat(-ori_camera_ee).matmul(
            (rotmat(-ori_ee_base).matmul(pos_marker_base - pos_ee_base)-pos_camera_ee))
    pos_marker_image = K.matmul(pos_marker_camera)
    return pos_marker_image[:2]/pos_marker_image[2]


def pointloss(param, obs_marker_2d, pos_ee_base, ori_ee_base, K, proj_func):
    proj_marker_2d = proj_func(param, pos_ee_base, ori_ee_base, K)
    return (obs_marker_2d - proj_marker_2d).norm()



def mean_loss(data, param, K, proj_func=hand_marker_proj_world_camera):
    losses = []
    for d in data:
        corner = d[0]
        ee_base_pos = d[1]
        ee_base_ori = d[2]
        ploss = pointloss(param, corner, ee_base_pos, ee_base_ori, K, proj_func)
        losses.append(ploss)
    return torch.stack(losses).mean()

def find_parameter(param, L):
    optimizer=torch.optim.LBFGS([param], max_iter=1000, lr=1, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=L(param)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return param.detach()


def sim_data(n, K, noise_std=0):
    from torchcontrol.transform import Rotation as R
    from torchcontrol.transform import Transformation as T
    #  z                 marker
    #  ^                /
    #  |               / 
    # (q0)----L---(q1)ee
    #  |
    #  H
    #  |                    > camera
    #  +--------------------+--->x
    #    
    L=0.3
    D=2.0 #camera-distance to robot base
    H=0.2
    get_ee = lambda q: torch.DoubleTensor([L * math.cos(q), L * math.sin(q), H])
    p_marker_0=torch.DoubleTensor([0., 0., 0.2]) #marker position on ee frame
    T_camera_ee = T.from_rot_xyz(
                rotation=R.from_rotvec(torch.DoubleTensor([-math.pi/2, 0, 0])) * R.from_rotvec(torch.DoubleTensor([0, -math.pi/2, 0])),  # camera orientation
                translation=torch.DoubleTensor([D, 0., 0.]))  # camera position
    gt_param = torch.cat([T_camera_ee.rotation().as_rotvec(), T_camera_ee.translation(), p_marker_0])
    data=[]
    for i in range(n):
            q0 = i * 2* math.pi / n
            q1 = i * 2* math.pi / n * 2
            ee = get_ee(q0)
            obs_Tee_base = T.from_rot_xyz(
                    rotation=(R.from_rotvec(torch.DoubleTensor([0., 0., q1])) * R.from_rotvec(torch.DoubleTensor([q1, 0., 0.]))),
                    translation=ee)
            p_marker_camera = (T_camera_ee.inv() * obs_Tee_base).apply(p_marker_0)
            p_marker_proj = K.matmul(p_marker_camera)
            p_marker_image = p_marker_proj[:2] / p_marker_proj[2]
            data.append([
                p_marker_image + torch.randn_like(p_marker_image) * noise_std,
                obs_Tee_base.translation(),
                obs_Tee_base.rotation().as_rotvec()])
            # print('q', [q0, q1],  '\n'
            #     'ee', ee, '\n'
            #     'marker_camera', p_marker_camera, '\n',
            #     'marker_image', p_marker_image)
    
    return data, gt_param


"""
Taken from https://stackoverflow.com/a/59086818/9654319
"""

from numpy.linalg import det
from scipy.stats import dirichlet
from scipy.spatial import ConvexHull, Delaunay
import numpy as np


def dist_in_hull(points, n):
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    result = np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))

    # lexsort sorts using last key as primary, then second-to-last, etc.
    sorted_indices = np.lexsort([result[:, 2], result[:, 1], result[:, 0]])
    return result[sorted_indices]
