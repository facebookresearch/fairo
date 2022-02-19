#!/usr/bin/env python
from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
# from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Transformation as T

import torch
import math
import scipy.optimize as opt
import numpy as np
from numpy.linalg import norm
np.set_printoptions(linewidth=200)


#https://polymetis-docs.github.io/torchcontrol-doc.html


# ex: 640x480 res https://github.com/facebookresearch/fairo/blob/austinw/artags/perception/fairotag/tutorials/data/realsense_intrinsics.json
def build_proj_matrix(fx, fy, ppx, ppy, coeff=None):
    # consider handle distortion here
    return torch.DoubleTensor([[fx, 0., ppx],
                                [0., fy, ppy],
                                [0., 0.,  1.]])



def loss(param, obs_marker_2d, obs_Tee_base, K):
    #rx, ry, rz, tx, ty, tz, px, py, pz  = param
    Tcamera_base = T.from_rot_xyz(
            rotation=R.from_rotvec(param[:3]),
            translation=param[3:6])
    p_marker_ee = param[6:9]

    Ttotal = (Tcamera_base.inv()  * obs_Tee_base)
    p_marker_camera = Ttotal.apply(p_marker_ee)
    p_marker_image = K.matmul(p_marker_camera)

    return (obs_marker_2d - p_marker_image[:2]/p_marker_image[2]).norm().item()


def data_loss(param, obs, K):
    err = np.ndarray(len(obs), dtype=np.float64)
    for i, (obs_marker_2d, obs_Tee_base) in enumerate(obs):
        err[i] = loss(param, obs_marker_2d, obs_Tee_base, K)
    return err

def sim_data(n, K):
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
                p_marker_image,
                obs_Tee_base])
            # print('q', [q0, q1],  '\n'
            #     'ee', ee, '\n'
            #     'marker_camera', p_marker_camera, '\n',
            #     'marker_image', p_marker_image)
    
    return data, gt_param



@torch.no_grad()
def test_opt(obs_data, gt_param):
    # testing data and answer
    K = build_proj_matrix(fx=613.9306030273438,  fy=614.3072713216146, ppx=322.1438802083333, ppy=241.59906514485678)
    obs_data, gt_param = sim_data(n=10, K=K)

    # loss function
    L = lambda p: data_loss (torch.tensor(p), obs_data, K)

    print('gt_param', gt_param)

    # verify optimality of true answer
    print(L(gt_param))
    assert norm(L(gt_param)) < 1e-2
    assert norm(L(gt_param)) < norm(L(gt_param + 0.0000001))
    assert norm(L(gt_param)) < norm(L(gt_param - 0.0000001))

    # least squares
    param0 = gt_param + 0.1*np.random.randn(*gt_param.shape)
    print('initial param', param0)

    # diff_step is needed to keep stepsize small that findiff  remain accurate
    res = opt.least_squares(L, param0, diff_step=0.000001) 
    pstar = res.x
    print('pstar', pstar) 
    print('L(pstar)', norm(L(pstar)))
    
    #compare lsq result with answer
    np.testing.assert_allclose(pstar, gt_param, atol=1e-4)


def plotdata(obs_data):
    import matplotlib.pyplot as plt


    c = np.arange(len(obs_data))
    plt.figure()
    ax=plt.subplot(1,2,1)
    ax.scatter([p[0].item()  for p,_ in obs_data],
             [p[1].item()  for p,_ in obs_data],
             s=0.1,
             c=c,
             cmap='hsv')

    ax.axis('image')
    ax.set_xlim([0, 640])
    ax.set_ylim([480, 0])

    ax.set_title('camera view')

    ax=plt.subplot(1,2,2, projection='3d')
    ax.scatter(
        [t.translation()[0].item() for _,t in obs_data],
        [t.translation()[1].item() for _,t in obs_data],
        [t.translation()[2].item() for _,t in obs_data],
        s=0.2,
        c=c,
        cmap='hsv'
    )
    ax.set_title('EE in base coordindate')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()
    plt.savefig('testdata.pdf')


def main():
    K = build_proj_matrix(fx=613.9306030273438,  fy=614.3072713216146, ppx=322.1438802083333, ppy=241.59906514485678)
    obs_data, gt_param = sim_data(n=100, K=K)
    plotdata(obs_data)
    test_opt(obs_data, gt_param)

main()
