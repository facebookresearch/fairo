import string
from dataclasses import dataclass

import numpy as np
import sophus as sp
import gtsam


DEFAULT_HUBER_C = 1.345


def sophus2gtsam(pose):
    return gtsam.Pose3(pose.matrix())


def gtsam2sophus(pose):
    return sp.SE3(pose.matrix())


def shorthand(idx):
    return getattr(gtsam.symbol_shorthand, string.ascii_uppercase[idx])


@dataclass
class ObjectInfo:
    t_valid: list
    is_static: bool


class FactorGraph:
    """Wrapper around GTSAM factor graph operations"""

    def __init__(self):
        self.objects = []

        # Noise
        self.zero_pose_noise = gtsam.noiseModel.Constrained.All(6)
        self.huber_noise = (gtsam.noiseModel.mEstimator.Huber(DEFAULT_HUBER_C),)

        # Initialize graph
        self.reset()

    def _get_symbol(self, idx, t):
        obj_info = self.objects[idx]
        if obj_info.is_static:
            t = 0

        return shorthand(idx)(t)

    def _get_latest_symbol(self, idx):
        obj_info = self.objects[idx]
        return self._get_symbol(idx, obj_info.t_valid[-1])

    def _get_prev_symbol(self, idx, t):
        obj_info = self.objects[idx]
        if t in obj_info.t_valid:
            return self._get_symbol(idx, t)
        else:
            return None

    def _get_obj_pose(self, idx, t=None):
        if t is None:
            symbol = self._get_latest_symbol(idx)
        else:
            symbol = self._get_symbol(idx, t)

        if symbol is not None:
            return self.estimate.atPose3(symbol)
        else:
            return None

    # Graph construction/update
    def reset(self):
        # Graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.estimate = gtsam.Values()
        self.new_estimate = gtsam.Values()

        # Tracking
        self.t = 0
        self.isam = gtsam.NonlinearISAM()

        # Objects
        for idx, obj_info in enumerate(self.objects):
            obj_info.t_valid = []
            symbol = self._get_symbol(idx, self.t)
            self.new_estimate.insert(symbol, gtsam.Pose3())

    def add_obj(self, is_static=False):
        # Update object info
        idx = len(self.objects)
        self.objects.append(ObjectInfo([], is_static))

        # Initialize estimate value
        t = 0 if is_static else self.t
        symbol = shorthand(idx)(t)
        self.new_estimate.insert(symbol, gtsam.Pose3())

        return idx

    def set_obj_pose(self, idx, pose):
        gt_pose = sophus2gtsam(pose)
        symbol = self._get_symbol(idx, self.t)
        if self.new_estimate.exists(symbol):
            self.new_estimate.update(symbol, gt_pose)

    def fix_obj_pose(self, idx, pose=None):
        if pose is None:
            gt_pose = self._get_obj_pose(idx)
        else:
            gt_pose = sophus2gtsam(pose)
        symbol = self._get_symbol(idx, self.t)

        # Add prior to graph
        noise = gtsam.noiseModel.Constrained.All(6)
        factor = gtsam.PriorFactorPose3(symbol, gt_pose, noise)
        self.graph.push_back(factor)

    def get_obj_pose(self, idx, t=None):
        return gtsam2sophus(self._get_obj_pose(idx, t))

    def add_observation(self, idx0, idx1, pose, noise, trans_noise=None):
        # Identify objects
        symbol0 = self._get_symbol(idx0, self.t)
        symbol1 = self._get_symbol(idx1, self.t)
        symbol_prev0 = self._get_prev_symbol(idx0, self.t - 1)
        symbol_prev1 = self._get_prev_symbol(idx1, self.t - 1)

        # Observation factor
        gt_pose = sophus2gtsam(pose)
        pose_noise = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(DEFAULT_HUBER_C),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(noise)),
        )
        factor = gtsam.BetweenFactorPose3(symbol0, symbol1, gt_pose, pose_noise)
        self.graph.push_back(factor)

        # Transition factor
        if trans_noise is not None:
            pose_noise = gtsam.noiseModel.Robust(
                gtsam.noiseModel.mEstimator.Huber(DEFAULT_HUBER_C),
                gtsam.noiseModel.Diagonal.Sigmas(np.array(trans_noise)),
            )

            for symbol, symbol_prev in [(symbol0, symbol_prev0), (symbol1, symbol_prev1)]:
                if symbol_prev is not None:
                    factor = gtsam.BetweenFactorPose3(
                        symbol_prev, symbol, gtsam.Pose3(), pose_noise
                    )
                    self.graph.push_back(factor)

    # Solver
    def optimize(self, verbosity=0):
        # Setup optimization
        params = gtsam.LevenbergMarquardtParams()
        if verbosity == 0:
            params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.new_estimate, params)

        # Optimize
        if verbosity > 0:
            print("Optimizing extrinsics...")
        self.new_estimate = optimizer.optimize()
        if verbosity > 0:
            print(f"initial error = {self.graph.error(self.init_values)}")
            print(f"final error = {self.graph.error(result)}")

    def increment(self, full_optimization=False):
        if self.t == 0 or full_optimization:
            self.optimize()

        # Incremental update
        self.isam.update(self.graph, self.new_estimate)
        self.estimate = self.isam.estimate()

        """
        # Update estimate
        for idx, obj_info in enumerate(self.objects):
            symbol = self._get_latest_symbol(idx)
            if not self.estimate.exists(symbol):
                symbol_prev = self._get_prev_symbol(idx)
                if self.t == 0:
                    pose = gtsam.Pose3()
                else:
                    pose = self.estimate.atPose3(symbol_prev)
                self.estimate.insert(symbol, pose)
        """

        # Update object infos
        for idx, obj_info in enumerate(self.objects):
            symbol = self._get_symbol(idx, self.t)
            if self.estimate.exists(symbol):
                obj_info.t_valid.append(self.t)

        # Increment timestep
        self.t += 1
        self.graph.resize(0)
        self.new_estimate = gtsam.Values()

        # Initialize estimates with previous values
        for idx, obj_info in enumerate(self.objects):
            if not obj_info.is_static:
                symbol = self._get_symbol(idx, self.t)
                symbol_prev = self._get_latest_symbol(idx)
                pose = self.estimate.atPose3(symbol_prev)
                self.new_estimate.insert(symbol, pose)

    # Save/load
    def save_graph(self):
        pass

    def load_graph(self, graph):
        pass
