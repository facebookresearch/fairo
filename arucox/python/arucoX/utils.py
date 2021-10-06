from typing import Tuple

import numpy as np
import sophus as sp
import gtsam
from scipy.spatial.transform import Rotation as R


def so3_to_quat(r: sp.SO3) -> np.ndarray:
    return R.from_matrix(r.matrix()).as_quat()


def se3_to_xyz_quat(t: sp.SE3) -> Tuple[np.ndarray, np.ndarray]:
    xyz = t.translation()
    quat = R.from_matrix(t.rotationMatrix()).as_quat()
    return xyz, quat


def sophus2gtsam(pose):
    return gtsam.Pose3(pose.matrix())


def gtsam2sophus(pose):
    return sp.SE3(pose.matrix())
