# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis.utils.test_policies import test_parametrized_data


inputs = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}

policies = [
    policy_class(**policy_kwargs)
    for (
        policy_class,
        policy_kwargs,
        is_terminating,
        update_params,
    ) in test_parametrized_data
]
scripted_policies = [torch.jit.script(policy) for policy in policies]


def time_policy_performance(policy):
    with torch.no_grad():
        policy.forward(inputs)


time_policy_performance.params = policies


def time_scripted_policy_performance(param):
    with torch.no_grad():
        param.forward(inputs)


time_scripted_policy_performance.params = scripted_policies
