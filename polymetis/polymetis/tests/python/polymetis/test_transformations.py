# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import torch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rs

from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R


def standardize_quat(q):
    if q[3] > 0:
        return q
    else:
        return -q


@pytest.fixture(params=range(10), autouse=True)
def set_seed(request):
    random.seed(request.param)
    np.random.seed(request.param)
    torch.manual_seed(request.param)


class TestRotation:
    def test_creation_conversion(self):
        """
        Tests conversions between quaternion, rotation matrix, and rotation vector
        Checks against scipy.spatial.transforms.Rotation
        """
        r_scipy = Rs.random()
        q_scipy = standardize_quat(r_scipy.as_quat())
        m_scipy = r_scipy.as_matrix()
        rv_scipy = r_scipy.as_rotvec()

        rt_q = R.from_quat(torch.Tensor(q_scipy))
        assert np.allclose(standardize_quat(rt_q.as_quat()), q_scipy, atol=1e-3)
        assert np.allclose(rt_q.as_matrix(), m_scipy, rtol=1e-3)
        assert np.allclose(rt_q.as_rotvec(), rv_scipy, rtol=1e-3)

        rt_m = R.from_matrix(torch.Tensor(m_scipy))
        assert np.allclose(standardize_quat(rt_m.as_quat()), q_scipy, atol=1e-3)
        assert np.allclose(rt_m.as_matrix(), m_scipy, rtol=1e-3)
        assert np.allclose(rt_m.as_rotvec(), rv_scipy, rtol=1e-3)

        rt_rv = R.from_rotvec(torch.Tensor(rv_scipy))
        assert np.allclose(standardize_quat(rt_rv.as_quat()), q_scipy, atol=1e-3)
        assert np.allclose(rt_rv.as_matrix(), m_scipy, rtol=1e-3)
        assert np.allclose(rt_rv.as_rotvec(), rv_scipy, rtol=1e-3)

    def test_operations(self):
        """
        Check consistency & correctness of rotation operations:
            inv, apply, __mul__, identity
        """
        r1_scipy = Rs.random()
        q1_scipy = standardize_quat(r1_scipy.as_quat())
        r1 = R.from_quat(torch.Tensor(q1_scipy))
        r2_scipy = Rs.random()
        q2_scipy = standardize_quat(r2_scipy.as_quat())
        r2 = R.from_quat(torch.Tensor(q2_scipy))

        v = torch.rand(3)

        # inv
        assert np.allclose(
            standardize_quat(r1.inv().as_quat()),
            standardize_quat(r1_scipy.inv().as_quat()),
            atol=1e-3,
        )

        # apply
        assert np.allclose(r1.apply(v), r1_scipy.apply(v.numpy()), rtol=1e-3)

        # __mul__
        assert np.allclose(
            standardize_quat((r1 * r2).as_quat()),
            standardize_quat((r1_scipy * r2_scipy).as_quat()),
            atol=1e-3,
        )

        # apply + __mul__
        assert np.allclose((r1 * r2).apply(v), r1.apply(r2.apply(v)), rtol=1e-3)

        # inv + __mul__ + identity
        assert np.allclose(
            standardize_quat((r1 * r1.inv()).as_quat()),
            standardize_quat(R.identity().as_quat()),
            atol=1e-3,
        )

    def test_axis_magnitude(self):
        """
        Tests if axis() is unit vector
        Tests if axis & magnitude is consistent with as_rotvec
        """
        r_scipy = Rs.random()
        r = R.from_quat(torch.Tensor(r_scipy.as_quat()))
        assert np.allclose(torch.norm(r.axis()), torch.ones(1), rtol=1e-7)
        assert np.allclose((r.axis() * r.magnitude()), r.as_rotvec(), rtol=1e-7)


class TestTransformation:
    def test_operations(self):
        """
        Check consistency & correctness of transformation operations:
            inv, apply, __mul__, identity
        """
        r1_scipy = Rs.random()
        q1_scipy = standardize_quat(r1_scipy.as_quat())
        r1 = R.from_quat(torch.Tensor(q1_scipy))
        p1 = torch.rand(3)
        t1 = T.from_rot_xyz(rotation=r1, translation=p1)

        r2_scipy = Rs.random()
        q2_scipy = standardize_quat(r2_scipy.as_quat())
        r2 = R.from_quat(torch.Tensor(q2_scipy))
        p2 = torch.rand(3)
        t2 = T.from_rot_xyz(rotation=r2, translation=p2)

        v = torch.rand(3)

        # apply()
        assert np.allclose(
            t1.apply(v), r1_scipy.apply(v.numpy()) + p1.numpy(), rtol=1e-3
        )

        # apply + __mul__
        assert np.allclose((t1 * t2).apply(v), t1.apply(t2.apply(v)), rtol=1e-3)

        # inv + __mul__ + identity
        assert np.allclose(
            (t1 * t1.inv()).as_matrix(), T.identity().as_matrix(), atol=1e-3
        )
