from typing import Optional, List
from functools import partial

import numpy as np
import sophus as sp
import gtsam

import fairotag as frt
from arucoX.utils import sophus2gtsam, gtsam2sophus


POS_NOISE = 0.001
ORI_NOISE = 0.005
NUM_SAMPLES = 150


def pose_jacobian_numerical(f, x, delta=1e-8):
    jac = np.zeros([6, 6])
    for i in range(6):
        delta_arr = np.zeros(6)
        delta_arr[i] = delta
        pose_offset_p = gtsam.Pose3.Expmap(delta_arr) * x
        pose_offset_n = gtsam.Pose3.Expmap(-delta_arr) * x
        jac[:, i] = (f(pose_offset_p) - f(pose_offset_n)) / (2 * delta)

    return jac


def pose_jacobian_analytical(f, x):
    jac = np.zeros([6, 6])
    jac_x = gtsam.Pose3.LogmapDerivative(x)
    for i in range(6):
        jac[:, i] = f(gtsam.Pose3.Expmap(jac_x[:, i]))

    return jac


def frame_error_func(this: gtsam.CustomFactor, v, H: Optional[List[np.ndarray]]):
    pose0 = v.atPose3(this.keys()[0])
    pose1 = v.atPose3(this.keys()[1])
    pose2 = v.atPose3(this.keys()[2])

    def pose_err(pose_0, pose_1, pose_01_expected):
        pose_01 = pose_0.between(pose_1)
        error = pose_01_expected.localCoordinates(pose_01)
        return error

    error = pose_err(pose0, pose1, pose2)

    if H is not None:

        def get_jacobians(jac_func):
            H = [None for _ in range(3)]
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

            return H

        H_analytical = get_jacobians(pose_jacobian_analytical)
        H_numerical = get_jacobians(pose_jacobian_numerical)

        for i in range(3):
            # print("-------------")
            # print(H_analytical[i], H_numerical[i], H_analytical[i] - H_numerical[i])
            # assert np.allclose(H_analytical[i], H_numerical[i], atol=1e-3, rtol=1e-2)
            H[i] = H_numerical[i]
            # H[i] = H_analytical[i]

    return error


class Graph:
    """
    0: robot base frame
    1: static marker frame
    2: end-effector frame
    3: end-effector marker frame
    """

    def __init__(self):
        self.fgraph = gtsam.NonlinearFactorGraph()
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        )
        self.zero_noise = gtsam.noiseModel.Constrained.All(6)

        self.values = gtsam.Values()
        self.B = gtsam.symbol_shorthand.B  # static objects
        self.M = gtsam.symbol_shorthand.M  # moving object 1
        self.N = gtsam.symbol_shorthand.N  # moving object 2
        self.T = gtsam.symbol_shorthand.T  # transforms

        # Base object prior factor & init value
        factor = gtsam.PriorFactorPose3(self.B(0), gtsam.Pose3(), self.zero_noise)
        self.fgraph.push_back(factor)

        self.values.insert(self.B(0), gtsam.Pose3())
        self.values.insert(self.B(1), gtsam.Pose3())
        self.values.insert(self.T(0), gtsam.Pose3())
        self.values.insert(self.T(1), gtsam.Pose3())

        # Obs count
        self.i_obs = 0

    def add_obs(self, t02_obs, t13_obs, t03_truth):
        t02_obs_g = sophus2gtsam(t02_obs)
        t13_obs_g = sophus2gtsam(t13_obs)

        # Add between factors
        factor = gtsam.BetweenFactorPose3(
            self.B(0), self.M(self.i_obs), t02_obs_g, self.pose_noise
        )
        self.fgraph.push_back(factor)
        factor = gtsam.BetweenFactorPose3(
            self.B(1), self.N(self.i_obs), t13_obs_g, self.pose_noise
        )
        self.fgraph.push_back(factor)

        # Add same-frame factors
        if self.i_obs == 0:
            factor = gtsam.CustomFactor(
                self.pose_noise, [self.B(0), self.B(1), self.T(0)], frame_error_func
            )
            self.fgraph.push_back(factor)
        factor = gtsam.CustomFactor(
            self.pose_noise, [self.M(self.i_obs), self.N(self.i_obs), self.T(1)], frame_error_func
        )
        self.fgraph.push_back(factor)

        # Initialize values
        self.values.insert(self.M(self.i_obs), t02_obs_g)
        self.values.insert(self.N(self.i_obs), t02_obs_g)
        # self.values.insert(self.N(self.i_obs), sophus2gtsam(t03_truth)) #ground truth

        # Increment count
        self.i_obs += 1

    def optimize(self, verbosity=0):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity(["SILENT", "TERMINATION"][verbosity])
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.fgraph, self.values, params)

        self.values = optimizer.optimize()

    def get_transforms(self):
        t01_graph = gtsam2sophus(self.values.atPose3(self.T(0)))
        t23_graph = gtsam2sophus(self.values.atPose3(self.T(1)))

        return t01_graph, t23_graph


class NewGraphWrapper:
    def __init__(self):
        from arucoX.graph import FactorGraph

        self.graph = FactorGraph()

        # Noise
        self.pose_noise = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])

        # Initialize graph
        self.graph.init_variable("base", sp.SE3())
        self.graph.init_variable("base_marker", sp.SE3())
        self.graph.init_variable("base_transform", sp.SE3())
        self.graph.init_variable("ee_transform", sp.SE3())

        self.graph.add_prior("base", sp.SE3())
        self.graph.add_fixed_transform(
            "base", "base_marker", "base_transform", noise=self.pose_noise / NUM_SAMPLES
        )

        # Count
        self.n_snapshots = 0

        # Initialize results
        self.results = {}

    def add_obs(self, t02_obs, t13_obs, t03_truth=None):
        ee_name = f"ee_{self.n_snapshots}"
        ee_marker_name = f"ee_marker_{self.n_snapshots}"
        self.n_snapshots += 1

        # Introduce new variables
        self.graph.init_variable(ee_name, t02_obs)
        self.graph.init_variable(ee_marker_name, t02_obs)

        # Add new information
        self.graph.add_observation("base", ee_name, t02_obs, noise=self.pose_noise)
        self.graph.add_observation("base_marker", ee_marker_name, t13_obs, noise=self.pose_noise)
        self.graph.add_fixed_transform(
            ee_name, ee_marker_name, "ee_transform", noise=self.pose_noise
        )

    def optimize(self, verbosity=0):
        self.results = self.graph.optimize(verbosity=verbosity)

    def get_transforms(self):
        return self.results["base_transform"], self.results["ee_transform"]


class Env:
    """
    0: robot base frame
    1: static marker frame
    2: end-effector frame
    3: end-effector marker frame

    Transform between 0 and 1 are fixed
    Transform between 2 and 3 are fixed
    `sample_obs` Samples different end-effector poses
    """

    def __init__(self):
        # Randomly generate 2 sets of objects in same frame
        # (Note: have 2 and 3 relatively close)
        self.t01 = sp.SE3.exp(np.random.randn(6))
        self.t23 = sp.SE3.exp(np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]) * np.random.randn(6))

        # Sample noise
        self.sample_hi = np.array([0.5, 0.5, 0.5, np.pi / 2, np.pi / 2, np.pi / 2])
        self.sample_lo = np.array([0.0, 0.0, 0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])

        # Obs noise (TODO: Add noise when working)
        self.noise = np.concatenate([POS_NOISE * np.ones(3), ORI_NOISE * np.ones(3)])

    def get_transforms(self):
        return self.t01, self.t23

    def sample_obs(self):
        # Move 2 to random pose (sampled from uniform)
        t2 = sp.SE3.exp(np.random.uniform(low=self.sample_lo, high=self.sample_hi))

        # Compute pose of 1 & 3
        t1 = self.t01
        t3 = t2 * self.t23

        # Compute observations (from 0 to 2 & from 1 to 3)
        t02 = t2
        t13 = t1.inverse() * t3

        # Add noise to observations
        t02 = t02 * sp.SE3.exp(self.noise * np.random.randn(6))
        t13 = t13 * sp.SE3.exp(self.noise * np.random.randn(6))

        return t02, t13


if __name__ == "__main__":
    # Initialize
    # graph = Graph()
    graph = NewGraphWrapper()
    env = Env()

    # Sample & add observations
    for _ in range(NUM_SAMPLES):
        t02_obs, t13_obs = env.sample_obs()
        graph.add_obs(t02_obs, t13_obs, t02_obs * env.t23)

    # Optimize
    graph.optimize(verbosity=1)

    # Compare results
    t01_env, t23_env = env.get_transforms()
    t01_graph, t23_graph = graph.get_transforms()

    print("=== origin2base ===")
    print(f"Env: {t01_env}")
    print(f"Graph: {t01_graph}")
    print(f"error={(t01_env * t01_graph.inverse()).log()}")

    print("=== ee2marker ===")
    print(f"Env: {t23_env}")
    print(f"Graph: {t23_graph}")
    print(f"error={(t23_env * t23_graph.inverse()).log()}")
