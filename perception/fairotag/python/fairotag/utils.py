from typing import Tuple, Dict

import numpy as np
import sophus as sp
import gtsam
from scipy.spatial.transform import Rotation as R

import fairotag as frt


def so3_to_quat(r: sp.SO3) -> np.ndarray:
    return R.from_matrix(r.matrix()).as_quat()


def se3_to_xyz_quat(t: sp.SE3) -> Tuple[np.ndarray, np.ndarray]:
    xyz = t.translation()
    quat = R.from_matrix(t.rotationMatrix()).as_quat()
    return xyz, quat


def xyz_quat_to_se3(pos: np.ndarray, quat: np.ndarray) -> sp.SE3:
    return sp.SE3(R.from_quat(quat).as_matrix(), pos)


def sophus2gtsam(pose):
    return gtsam.Pose3(pose.matrix())


def gtsam2sophus(pose):
    return sp.SE3(pose.matrix())


def intrinsics2dict(intrinsics):
    return {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "coeffs": list(intrinsics.coeffs),
    }


def dict2intrinsics(intrinsics_dict: Dict):
    return frt.CameraIntrinsics(
        fx=intrinsics_dict["fx"],
        fy=intrinsics_dict["fy"],
        ppx=intrinsics_dict["ppx"],
        ppy=intrinsics_dict["ppy"],
        coeffs=intrinsics_dict["coeffs"],
    )
