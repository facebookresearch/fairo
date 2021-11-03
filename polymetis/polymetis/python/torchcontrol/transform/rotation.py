# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import torch
from polymetis.utils.data_dir import PKG_ROOT_DIR


try:
    torch.ops.load_library(f"{os.environ['CONDA_PREFIX']}/lib/libtorchrot.so")
except OSError:
    lib_path = os.path.abspath(
        os.path.join(
            PKG_ROOT_DIR,
            "../../build/torch_isolation/libtorchrot.so",
        )
    )
    print(
        f"Warning: Failed to load 'libtorchrot.so' from CONDA_PREFIX, loading from default build directory instead: '{lib_path}'"
    )
    torch.ops.load_library(lib_path)

functional = torch.ops.torchrot


@torch.jit.script
class RotationObj:
    """
    A scriptable rotation object used for storing & manipulating rotations in 3D space.
    The API is similar to that of scipy.spatial.transform.Rotation.

    Rotation parameters are stored as quaternions due to:
        1. Efficient computational properties
        2. Compactness (compared to rotation matrices)
        3. Avoidance of gimbal lock (euler angles), ambiguity (axis-angle)

    Quaternions follow the convention of <x, y, z, w>.
    """

    def __init__(self, q: torch.Tensor):
        assert q.numel() == 4
        self._q = q

    def __repr__(self):
        return f"RotationObj(quaternion={self._q})"

    # Conversions
    def as_quat(self) -> torch.Tensor:
        """
        Returns:
            Quaternion representation of rotation
        """
        return self._q.clone()

    def as_matrix(self) -> torch.Tensor:
        """
        Returns:
            Matrix representation of rotation
        """
        return torch.ops.torchrot.quat2matrix(self._q)

    def as_rotvec(self) -> torch.Tensor:
        """
        Returns:
            Rotation vector representation of rotation
        """
        return torch.ops.torchrot.quat2rotvec(self._q)

    # Properties
    def axis(self) -> torch.Tensor:
        """
        Returns:
            Axis of rotation
        """
        return torch.ops.torchrot.quat2axis(self._q)

    def magnitude(self) -> torch.Tensor:
        """
        Returns:
            Magnitude of rotation
        """
        return torch.ops.torchrot.quat2angle(self._q)

    # Operations
    def apply(self, v):
        """Applies the rotation to a vector

        Args:
            v: Input vector of shape (3,)

        Returns:
            Resulting vector of shape (3,)
        """
        assert v.shape[-1] == 3
        return self.as_matrix() @ v

    def __mul__(self, r_other: RotationObj) -> RotationObj:
        """Stacks two rotations to form a new rotation
        Example: r_new = r_1 * r_2
        """
        q_result = torch.ops.torchrot.quaternion_multiply(self._q, r_other._q)
        return RotationObj(q_result)

    def inv(self) -> RotationObj:
        """Inverts the rotation

        Returns:
            Inverted RotationObj
        """
        q_result = torch.ops.torchrot.invert_quaternion(self._q)
        return RotationObj(q_result)


# Creation functions
def from_quat(quat: torch.Tensor) -> RotationObj:
    """Creates a rotation object from a quaternion

    Args:
        quat: Quaternion representation

    Returns:
        Resulting RotationObj
    """
    assert quat.shape == torch.Size([4]), f"Invalid quaternion shape: {quat.shape}"
    return RotationObj(torch.ops.torchrot.normalize_quaternion(quat))


def from_matrix(matrix: torch.Tensor) -> RotationObj:
    """Creates a rotation object from a rotation matrix

    Args:
        quat: Matrix representation

    Returns:
        Resulting RotationObj
    """
    assert matrix.shape == torch.Size(
        [3, 3]
    ), f"Invalid rotation matrix shape: {matrix.shape}"
    return RotationObj(torch.ops.torchrot.matrix2quat(matrix))


def from_rotvec(rotvec: torch.Tensor) -> RotationObj:
    """Creates a rotation object from a rotation vector

    Args:
        quat: Rotation vector representation

    Returns:
        Resulting RotationObj
    """
    assert rotvec.shape == torch.Size(
        [3]
    ), f"Invalid rotation vector shape: {rotvec.shape}"
    return RotationObj(torch.ops.torchrot.rotvec2quat(rotvec))


def identity() -> RotationObj:
    """Creates a zero rotation object

    Returns:
        Identity RotationObj
    """
    return RotationObj(torch.tensor([0.0, 0.0, 0.0, 1.0]))
