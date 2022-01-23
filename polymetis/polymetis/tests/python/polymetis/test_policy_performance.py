# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from polymetis.utils.test_policies import test_parametrized_data


perf_parametrized_data = [x for x in test_parametrized_data if not x[2]]
perf_parametrized_names = [x[0].__name__ for x in perf_parametrized_data]


@pytest.fixture(params=perf_parametrized_data, ids=perf_parametrized_names)
def policy(request):
    policy_class, policy_kwargs, is_terminating, update_params = request.param
    return policy_class(**policy_kwargs)


@pytest.mark.benchmark(group="non-scripted")
def test_policy_performance(policy, benchmark):
    inputs = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}

    with torch.no_grad():
        benchmark(policy.forward, inputs)


@pytest.mark.benchmark(group="scripted")
def test_scripted_policy_performance(policy, benchmark):
    inputs = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}
    scripted_policy = torch.jit.script(policy)

    with torch.no_grad():
        benchmark(scripted_policy.forward, inputs)
