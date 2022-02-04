# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis.utils.test_policies import test_parametrized_data


inputs = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}

perf_parametrized_data = [x for x in test_parametrized_data if not x[2]]
policy_info = [(x[0], x[1]) for x in perf_parametrized_data]


class TimePolicyPerformance:
    params = range(len(policy_info))

    def setup(self, i):
        policy_class, policy_args = policy_info[i]
        self.policy = policy_class(**policy_args)
        self.scripted_policy = torch.jit.script(self.policy)
        self.repeat = 20

    def time_policy_performance(self, i):
        with torch.no_grad():
            self.policy.forward(inputs)

    def time_scripted_policy_performance(self, i):
        with torch.no_grad():
            self.scripted_policy.forward(inputs)
