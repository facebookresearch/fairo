from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sophus as sp
import gtsam


DEFAULT_HUBER_C = 1.345
USE_ANALYTICAL_JACOBIANS = False


# Factor graph object
class FactorGraph:
    """Wrapper around gtsam.NonlinearFactorGraph that provides functionalities:
    - API that uses Sophus for pose inputs/outputs
    - Macro methods for adding common factors
    - Automatically maps named variables to gtsam shorthand variables
    """

    def __init__(self):
        self.gtsam_graph = gtsam.NonlinearFactorGraph()
        self.factor_edges = {}  # adjacency list (dict)

        # Variables
        self.values = gtsam.Values()
        self.vars = {}

        self.n_variables = 0

        # TODOs for incremental smoothing:
        # - maintain active values
        # - update/delete factors

    @staticmethod
    def _process_noise(noise):
        if noise is None:
            return gtsam.noiseModel.Constrained.All(6)
        else:
            noise_gt = np.concatenate([np.array(noise[:3]), np.array(noise[:3])])
            return gtsam.noiseModel.Robust(
                gtsam.noiseModel.mEstimator.Huber(DEFAULT_HUBER_C),
                gtsam.noiseModel.Diagonal.Sigmas(noise_gt),
            )

    def init_variable(self, name, pose=sp.SE3()):
        pose_gt = sophus2gtsam(pose)

        # Update pose only if already exists
        if name in self.vars:
            var = self.vars[name]
            self.values.update(var, pose_gt)
            return

        # Create symbol
        var = gtsam.symbol_shorthand.X(self.n_variables)
        self.vars[name] = var
        self.n_variables += 1

        # Add to values
        self.values.insert(var, pose_gt)

        # Add to edges
        self.factor_edges[name] = []

    def add_prior(self, var_name, transform, noise=None):
        """ Prior factor """
        noise_gt = self._process_noise(noise)
        transform_gt = sophus2gtsam(transform)

        var = self.vars[var_name]

        factor = gtsam.PriorFactorPose3(var, transform_gt, noise_gt)
        self.gtsam_graph.push_back(factor)

    def add_observation(self, var1_name, var2_name, transform, noise=None):
        """ Between factor """
        noise_gt = self._process_noise(noise)
        transform_gt = sophus2gtsam(transform)

        var1 = self.vars[var1_name]
        var2 = self.vars[var2_name]

        factor = gtsam.BetweenFactorPose3(var1, var2, transform_gt, noise_gt)
        self.gtsam_graph.push_back(factor)

        # Add edge information
        self.factor_edges[var1].append((var2, transform_gt))
        self.factor_edges[var2].append((var1, transform_gt.inverse()))

    def add_fixed_transform(self, var1_name, var2_name, transform_name, noise=None):
        """ Custom factor for constant transforms """
        noise_gt = self._process_noise(noise)

        var1 = self.vars[var1_name]
        var2 = self.vars[var2_name]
        transform = self.vars[transform_name]

        factor = gtsam.CustomFactor(noise_gt, [var1, var2, transform], frame_error_func)
        self.gtsam_graph.push_back(factor)

    def bfs_initialization(self, root_var_name):
        var0 = self.vars[root_var_name]

        queue = [(var0, self.values.atPose3(var0))]
        visited = set([var0])

        while queue:
            curr_var, pose = queue.pop(0)
            self.values.update(curr_var, pose)

            for next_var, transform in self.factor_edges[curr_var]:
                if next_var not in visited:
                    queue.append((next_var, pose * transform))
                    visited.add(curr_var)

    def optimize(self, verbosity=0):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity(["SILENT", "TERMINATION"][verbosity])
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.gtsam_graph, self.values, params)

        result_values = optimizer.optimize()

        return {name: gtsam2sophus(result_values.atPose3(var)) for name, var in self.vars.items()}


# Helper functions
def sophus2gtsam(pose):
    return gtsam.Pose3(pose.matrix())


def gtsam2sophus(pose):
    return sp.SE3(pose.matrix())


# Custom factor for frames
def pose_jacobian_numerical(f, x, delta=1e-5):
    jac = np.zeros([6, 6])
    for i in range(6):
        delta_arr = np.zeros(6)
        delta_arr[i] = delta
        pose_offset_p = gtsam.Pose3.Expmap(delta_arr) * x
        pose_offset_n = gtsam.Pose3.Expmap(-delta_arr) * x
        jac[:, i] = (f(pose_offset_p) - f(pose_offset_n)) / (2 * delta)

    return jac


def pose_jacobian_analytical(f, x):  # TODO: Debug
    jac = np.zeros([6, 6])
    jac_x = gtsam.Pose3.LogmapDerivative(x)
    for i in range(6):
        jac[:, i] = f(gtsam.Pose3.Expmap(jac_x[:, i]))

    return jac


def frame_error_func(this: gtsam.CustomFactor, v, H: Optional[List[np.ndarray]]):
    pose0 = v.atPose3(this.keys()[0])
    pose1 = v.atPose3(this.keys()[1])
    pose2 = v.atPose3(this.keys()[2])

    # Compute error
    def pose_err(pose_0, pose_1, pose_01_expected):
        pose_01 = pose_0.between(pose_1)
        error = pose_01_expected.localCoordinates(pose_01)
        return error

    error = pose_err(pose0, pose1, pose2)

    # Compute Jacobians
    if H is not None:
        if USE_ANALYTICAL_JACOBIANS:
            jac_func = pose_jacobian_analytical
        else:
            jac_func = pose_jacobian_numerical

        H[0] = jac_func(
            lambda x: pose_err(x, pose1, pose2),
            x=pose0,
        )
        H[1] = jac_func(
            lambda x: pose_err(pose0, x, pose2),
            x=pose1,
        )
        H[2] = jac_func(
            lambda x: pose_err(pose0, pose1, x),
            x=pose2,
        )

    return error
