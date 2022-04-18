import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import fairomsg
import capnp

import io
import numpy as np
import open3d
import graspnetAPI

std_msgs = fairomsg.get_msgs("std_msgs")
sensor_msgs = fairomsg.get_msgs("sensor_msgs")
geometry_msgs = fairomsg.get_msgs("geometry_msgs")

"""Byte conversions"""

def np_to_bytes(arr: np.ndarray):
    with io.BytesIO() as f:
        np.save(f, arr)
        return f.getvalue()


def bytes_to_np(bytes_arr: bytes):
    with io.BytesIO(bytes_arr) as f:
        return np.load(f)


def grasp_group_to_bytes(grasp_group: graspnetAPI.grasp.GraspGroup):
    return np_to_bytes(grasp_group.grasp_group_array)


def bytes_to_grasp_group(arr_bytes: bytes):
    return graspnetAPI.grasp.GraspGroup(bytes_to_np(arr_bytes))


def open3d_pcd_to_bytes(cloud: open3d.geometry.PointCloud):
    if cloud.has_colors():
        arr = np.hstack([np.asarray(cloud.points), np.asarray(cloud.colors)])
    else:
        arr = np.asarray(cloud.points)
    return np_to_bytes(arr)


def bytes_to_open3d_pcd(arr_bytes: bytes):
    arr = bytes_to_np(arr_bytes)
    result = open3d.geometry.PointCloud(
        open3d.cuda.pybind.utility.Vector3dVector(arr[:, :3])
    )
    if arr.shape[1] == 6:
        result.colors = open3d.cuda.pybind.utility.Vector3dVector(arr[:, 3:])

    return result

"""Capnp conversions"""

def pcd_to_capnp(pcd: o3d.geometry.PointCloud):
    result = sensor_msgs.PointCloud2()
    result.data = open3d_pcd_to_bytes(pcd)
    return result

def capnp_to_pcd(blob):
    capnp_pcd = sensor_msgs.PointCloud2.from_bytes(blob)
    return bytes_to_open3d_pcd(capnp_pcd.data)

def grasp_group_to_capnp(grasp_group: graspnetAPI.grasp.GraspGroup):
    capnp_gg = std_msgs.ByteMultiArray()
    capnp_gg.data = grasp_group_to_bytes(grasp_group)
    # capnp_gg = geometry_msgs.PoseArray()
    # capnp_poses = capnp_gg.init("poses", len(grasp_group))
    # for pose, grasp in zip(capnp_poses, grasp_group):
    #     x, y, z = grasp.translation.tolist()
    #     pose.position = geometry_msgs.Point(x=x, y=y, z=z)
    #     x, y, z, w = R.from_matrix(grasp.rotation_matrix).as_quat().tolist()
    #     pose.orientation = geometry_msgs.Quaternion(x=x, y=y, z=z, w=w)
    return capnp_gg

def capnp_to_grasp_group(blob):
    capnp_gg = std_msgs.ByteMultiArray.from_bytes(blob)

    # for pose in capnp_gg.poses:
    #     xyz = pose.position
    
    gg = bytes_to_grasp_group(capnp_gg.data)
    return gg
