import math
import numpy as np
import open3d as o3d


def to_o3d_rgbd(rgb, depth, compressed=False):
    # convert to open3d RGBDImage
    rgb_u8 = np.ascontiguousarray(rgb[:, :, [2, 1, 0]], dtype=np.uint8)
    if compressed:
        depth_f32 = np.ascontiguousarray(depth, dtype=np.float32)
    else:
        depth_f32 = np.ascontiguousarray(depth, dtype=np.float32) * 1000
    orgb = o3d.geometry.Image(rgb_u8)
    odepth = o3d.geometry.Image(depth_f32)
    orgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        orgb, odepth, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    return orgbd


def compute_base_extrinsic(base_xyt):
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, base_xyt[2]])
    translation = [
        base_xyt[0],
        base_xyt[1],
        0,
    ]
    transform = np.identity(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    extrinsic = np.linalg.inv(transform)
    return extrinsic


def compute_extrinsic(base_xyt, cam_to_base_transform):
    # create transform matrix
    roty90 = o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi / 2, 0])
    rotxn90 = o3d.geometry.get_rotation_matrix_from_axis_angle([-math.pi / 2, 0, 0])
    rotz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, base_xyt[2]])
    rot_cam = cam_to_base_transform[:3, :3]
    trans_cam = cam_to_base_transform[:3, 3]
    final_rotation = rotz @ rot_cam @ rotxn90 @ roty90
    final_translation = [
        trans_cam[0] + base_xyt[0],
        trans_cam[1] + base_xyt[1],
        trans_cam[2] + 0,
    ]
    final_transform = np.identity(4)
    final_transform[:3, :3] = final_rotation
    final_transform[:3, 3] = final_translation
    extrinsic = np.linalg.inv(final_transform)
    return extrinsic


def compute_pcd(
    rgb, depth, cam_transform, base_state, intrinsic, compressed=False, in_base_frame=False
):

    orgbd = to_o3d_rgbd(rgb, depth, compressed=compressed)
    if in_base_frame:
        extrinsic = compute_extrinsic(np.asarray([0.0, 0.0, 0.0]), cam_transform)
    else:
        extrinsic = compute_extrinsic(base_state, cam_transform)
    opcd = o3d.geometry.PointCloud.create_from_rgbd_image(orgbd, intrinsic, extrinsic)
    return opcd
