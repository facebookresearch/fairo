# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torchcontrol as toco


class InverseDynamics(toco.ControlModule):
    """Computes inverse dynamics"""

    use_grav_comp: bool

    def __init__(self, robot_model, ignore_gravity: bool = True):
        """
        Args:
            robot_model: A valid robot model module from torchcontrol.models
            ignore_gravity: Whether to ignore gravitational effects
        """
        super().__init__()
        self.robot_model = robot_model
        self.ignore_gravity = ignore_gravity

    def forward(
        self,
        q_current: torch.Tensor,
        qd_current: torch.Tensor,
        qdd_desired: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_current: Current generalized coordinates
            qd_current: Current generalized velocities
            qdd_desired: Desired generalized accelerations

        Returns:
            Required generalized forces
        """
        result = self.robot_model.inverse_dynamics(q_current, qd_current, qdd_desired)
        if self.ignore_gravity:
            result -= self.robot_model.inverse_dynamics(
                q_current, torch.zeros_like(q_current), torch.zeros_like(q_current)
            )

        return result


class Coriolis(toco.ControlModule):
    """Computes the Coriolis force"""

    def __init__(self, robot_model):
        """
        Args:
            robot_model: A valid robot model module from torchcontrol.models
        """
        super().__init__()
        self.robot_model = robot_model

    def forward(
        self, q_current: torch.Tensor, qd_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q_current: Current generalized coordinates
            qd_current: Current generalized velocities

        Returns:
            Coriolis forces
        """
        u_all = self.robot_model.inverse_dynamics(
            q_current, qd_current, torch.zeros_like(q_current)
        )
        u_grav = self.robot_model.inverse_dynamics(
            q_current, torch.zeros_like(q_current), torch.zeros_like(q_current)
        )
        return u_all - u_grav
