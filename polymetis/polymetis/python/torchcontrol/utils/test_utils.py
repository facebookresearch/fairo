# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess

import torch


project_root_dir = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.strip()
    .decode("ascii")
)


class FakeRobotModel(torch.nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.n = num_joints

    def compute_jacobian(self, q):
        return torch.eye(6, self.n)

    def forward_kinematics(self, q):
        return torch.ones(3), torch.Tensor([0, 0, 0, 1])

    def inverse_dynamics(self, q, qd, qdd):
        return torch.ones(self.n)


def record_or_compare(test_name, dict_of_tensors, atol=1e-05):
    """record_or_compare makes it simple to compare generated data.
    Inspired by Ruby's VCR package.
    :param test_name: test name which is used as the saved filename.
        Typically request.node.name, from PyTest fixture `request`.
    :param dict_of_arrays: maps from strings to torch tensors.

    If this is the first time we ran this test, we save the data using
    torch.save into the subdir `data`. For future runs, we load this
    saved data and compare it against the loaded data.
    This function should only call once per test, as we use the test_name
    as the data filename. To refresh the data, simply delete the previous
    data, run once, and commit the new data to the repo.
    """
    # Locate data file
    data_dirname = os.path.join(project_root_dir, "polymetis/polymetis/tests/data")
    if not os.path.exists(data_dirname):
        os.makedirs(data_dirname)

    filename = os.path.join(data_dirname, f"{test_name}.pt")

    # Compare if file exists, otherwise save data to file
    if os.path.exists(filename):
        assert os.path.isfile(filename)
        loaded_dict_of_tensors = torch.load(filename)
        for key in dict_of_tensors:
            assert key in loaded_dict_of_tensors
            assert torch.allclose(
                loaded_dict_of_tensors[key],
                dict_of_tensors[key],
                atol=atol,
            ), f"Value mismatch in {key}."
            print(f"In {test_name}, '{key}' is equal to loaded '{key}'")
    else:
        torch.save(dict_of_tensors, filename)
        print(f"Saved data for {test_name}")
