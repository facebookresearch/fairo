
from collections import namedtuple

import torch
import cv2

from .utils import quat2rotvec, build_proj_matrix, mean_loss, find_parameter, rotmat, proj_funcs


CalibrationResult = namedtuple('CalibrationResult',
                                field_names=['num_marker_seen', 'stage2_retry', 'pixel_error', 'param', 'proj_func'],
                                defaults=[None]*5)

                            
# helper function
def extract_obs_data_std(data, camera_index):
    obs_data_std = []
    for d in data:
        if d['corners'][camera_index] is not None:
            obs_data_std.append((
                torch.tensor(d['corners'][camera_index], dtype=torch.float64),
                d['pos'].double(),
                quat2rotvec(d['ori'].double())
            ))

    ic = data[0]['intrinsics'][camera_index]
    K=build_proj_matrix(
        fx=ic['fx'],
        fy=ic['fy'],
        ppx=ic['ppx'],
        ppy=ic['ppy'])
    return obs_data_std, K


def solveEyeHandCalibration(corner_data, proj_func_name, pixel_tolerance):
    """
        A camera is observing a point marker mounted on a arm 
        p_obs = K T_unknown T_obs p_marker

        p_marker The marker is at 

    """
    proj_func = proj_funcs[proj_func_name]

    num_of_camera=len(corner_data[0]['intrinsics'])
    cal_results = []
    for i in range(num_of_camera):
        print(f'Solve camera {i}/{num_of_camera} pose')
        obs_data_std, K = extract_obs_data_std(corner_data, i)
        print('number of images with keypoint', len(obs_data_std))
        if len(obs_data_std) < 3:
            print('too few keypoint found for this camera, skip this camera')
            cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std)))
            continue

        # stage 1 - assuming marker is attach to EE origin, solve camera pose first
        if proj_func_name == "hand_marker_proj_world_camera":
            p3d = torch.stack([p[1] for p in obs_data_std]).detach().numpy()
        elif proj_func_name == "world_marker_proj_hand_camera":
            p3d = torch.stack([rotmat(-p[2]).matmul(-p[1]) for p in obs_data_std]).detach().numpy()

        p2d = torch.stack([p[0] for p in obs_data_std]).detach().numpy()
        retval, rvec, tvec = cv2.solvePnP(p3d, p2d, K.numpy(), distCoeffs=None, flags=cv2.SOLVEPNP_SQPNP)
        rvec_cam = torch.tensor(-rvec.reshape(-1))
        tvec_cam = -rotmat(rvec_cam).matmul(torch.tensor(tvec.reshape(-1)))
        pixel_error = mean_loss(obs_data_std, torch.cat([rvec_cam, tvec_cam, torch.zeros(3)]), K, proj_func).item()
        print('stage 1 mean pixel error', pixel_error)

        # stage 2 - allow marker to move, joint optimize camera pose and marker
        max_stage2_retry = 10
        stage2_retry_count = 0
        
        while True :

            stage2_retry_count += 1
            if stage2_retry_count > max_stage2_retry:
                cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std),
                                                     stage2_retry=stage2_retry_count,
                                                     param=param_star,
                                                     pixel_error=pixel_error,
                                                     proj_func=proj_func_name))
                print('Maximum stage2 retry execeeded, bailing out')
                break

            marker_max_displacement = 0.1 #meter
            param=torch.cat([rvec_cam, tvec_cam, torch.randn(3)*marker_max_displacement]).clone().detach()
            param.requires_grad=True
            L = lambda param: mean_loss(obs_data_std, param, K, proj_func)
            try:
                param_star=find_parameter(param, L)
            except Exception as e:
                print(e)
                continue

            pixel_error = L(param_star).item()
            print('stage 2 mean pixel error', pixel_error)
            if pixel_error > pixel_tolerance:
                print(f"Try again {stage2_retry_count}/{max_stage2_retry} because of poor solution {pixel_error} > {pixel_tolerance}")
            else:
                print(f"Good solution {pixel_error} <= {pixel_tolerance}")
                cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std),
                                                     stage2_retry=stage2_retry_count,
                                                     param=param_star,
                                                     pixel_error=pixel_error,
                                                     proj_func=proj_func_name))
                break

    return cal_results