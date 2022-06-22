import numpy as np
import open3d as o3d
import fairomsg

import site
import os
import io
import numpy as np
import open3d
import graspnetAPI
import capnp
import polygrasp

import cv2

std_msgs = fairomsg.get_msgs("std_msgs")
sensor_msgs = fairomsg.get_msgs("sensor_msgs")
geometry_msgs = fairomsg.get_msgs("geometry_msgs")

print("capnp loading polygrasp")
_schema_parser = capnp.SchemaParser()
polygrasp_msgs = _schema_parser.load(
    os.path.join(polygrasp.__path__[0], "polygrasp.capnp"),
    imports=site.getsitepackages(),
)

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
    with sensor_msgs.PointCloud2.from_bytes(blob) as capnp_pcd:
        return bytes_to_open3d_pcd(capnp_pcd.data)


def grasp_group_to_capnp(grasp_group: graspnetAPI.grasp.GraspGroup):
    capnp_gg = std_msgs.ByteMultiArray()
    capnp_gg.data = grasp_group_to_bytes(grasp_group)
    return capnp_gg


def capnp_to_grasp_group(blob):
    with std_msgs.ByteMultiArray.from_bytes(blob) as capnp_gg:
        gg = bytes_to_grasp_group(capnp_gg.data)
        return gg


def rgbd_to_capnp(rgbd):
    img = sensor_msgs.Image()
    img.data = np_to_bytes(rgbd)

    return img


def capnp_to_rgbd(blob):
    with sensor_msgs.Image.from_bytes(blob) as img:
        return bytes_to_np(img.data)


def load_bw_img(path):
    grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    thresh, bw_img = cv2.threshold(
        grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return bw_img.astype(bool)


def save_bw_img(img, name):
    cv2.imwrite(f"{name}.png", img * 255)
