# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch

from . import rotation as R
from .rotation import RotationObj


@torch.jit.script
class TransformationObj:
    """
    A scriptable transformation object used for storing & manipulating transformations in 3D space.
    A transformation consists of a translation and a rotation.

    Quaternions follow the convention of <x, y, z, w>.
    """

    _r: RotationObj

    def __init__(self, rotation: RotationObj, translation: torch.Tensor):
        assert translation.numel() == 3
        self._r = rotation
        self._x = translation

    def __repr__(self):
        return f"TransformationObj(\n\trotation={self._r}, \n\ttranslation={self._x}\n)"

    # Conversions
    def as_matrix(self) -> torch.Tensor:
        """
        Returns:
            Matrix representation of transformation (4-by-4)
        """
        T = torch.eye(4)
        T[0:3, 0:3] = self._r.as_matrix()
        T[0:3, 3] = self._x
        return T

    def as_twist(self) -> torch.Tensor:
        return torch.cat([self._x, self._r.as_rotvec()])

    # Properties
    def rotation(self) -> RotationObj:
        """
        Returns:
            Rotation component as a RotationObj
        """
        return self._r

    def translation(self) -> torch.Tensor:
        """
        Returns:
            Translation component as a vector
        """
        return self._x.clone()

    # Operations
    def apply(self, v: torch.Tensor):
        """Applies the transformation to a vector
        tf.apply(v) = tf.rotation().apply(v) + tf.translation
        """
        assert v.shape[-1] == 3
        return self._r.apply(v) + self._x

    def __mul__(self, tf_other: TransformationObj) -> TransformationObj:
        """Stacks two transformations to form a new transformation
        Example: tf_new = tf_1 * tf_2
        """
        return TransformationObj(
            rotation=self._r * tf_other._r,
            translation=self._r.apply(tf_other._x) + self._x,
        )

    def inv(self) -> TransformationObj:
        """Inverts the transformation
        Returns:
            Inverted TransformationObj
        """
        r_inv = self._r.inv()
        return TransformationObj(rotation=r_inv, translation=-r_inv.apply(self._x))


# Creation functions
def from_rot_xyz(rotation: RotationObj, translation: torch.Tensor) -> TransformationObj:
    """Creates a translation object from a translation vector and a a rotation object

    Args:
        translation: Translation component as a vector
        rotation: Rotation component as a RotationObj

    Returns:
        Resulting TranslationObj
    """
    assert translation.shape == torch.Size([3])
    return TransformationObj(rotation=rotation, translation=translation)


def from_matrix(T: torch.Tensor) -> TransformationObj:
    """Creates a translation object from a translation vector and a rotation object

    Args:
        T: Transformation matrix representation

    Returns:
        Resulting TranslationObj
    """
    assert T.shape == torch.Size([4, 4])
    return TransformationObj(rotation=R.from_matrix(T[0:3, 0:3]), translation=T[0:3, 3])


def identity() -> TransformationObj:
    """Creates a zero transformation object

    Returns:
        Identity TransformationObject
    """
    return TransformationObj(rotation=R.identity(), translation=torch.zeros(3))
