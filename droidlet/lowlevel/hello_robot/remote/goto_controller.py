from typing import List, Optional
import time
import threading

import numpy as np
import rospy

from utils import transform_global_to_base, transform_base_to_global

V_MAX_DEFAULT = 0.2  # base.params["motion"]["default"]["vel_m"]
W_MAX_DEFAULT = 0.45  # (vel_m_max - vel_m_default) / wheel_separation_m
ACC_LIN = 0.1  # 0.25 * base.params["motion"]["max"]["accel_m"]
ACC_ANG = 0.3  # 0.25 * (accel_m_max - accel_m_default) / wheel_separation_m


class GotoVelocityController:
    def __init__(
        self,
        robot,
        hz: float,
        v_max: Optional[float] = None,
        w_max: Optional[float] = None,
    ):
        self.robot = robot
        self.hz = hz
        self.dt = 1.0 / self.hz

        # Params
        self.v_max = v_max or V_MAX_DEFAULT
        self.w_max = w_max or W_MAX_DEFAULT
        self.lin_error_tol = 2 * self.v_max / hz
        self.ang_error_tol = 2 * self.w_max / hz

        # Initialize
        self.loop_thr = None
        self.control_lock = threading.Lock()
        self.active = False

        self.xyt_loc = self.robot.get_estimator_pose()
        self.xyt_goal = self.xyt_loc
        self.track_yaw = True

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max, tol=0.0, use_acc=True):
        """
        Computes velocity based on distance from target.
        Used for both linear and angular motion.

        Current implementation: Trapezoidal velocity profile
        """
        if use_acc:
            t = np.sqrt(2.0 * max(abs(x_err) - tol, 0.0) / a)  # x_err = (1/2) * a * t^2
            v = min(a * t, v_max)
            return v * np.sign(x_err)
        else:
            return np.sign(x_err) * (abs(x_err) > tol) * v_max

    @staticmethod
    def _projection_velocity_multiplier(theta_diff, tol=0.0):
        """
        Compute velocity muliplier based on yaw (faster if facing towards target).
        Used to control linear motion.

        Current implementation:
        Output = 1 when facing target, gradually decreases to 0 when angle to target is pi/3.
        """
        assert theta_diff >= 0.0
        return 1.0 - np.sin(max(theta_diff - tol, 0.0))

    @staticmethod
    def _turn_rate_limit(lin_err, heading_diff, w_max, dead_zone=0.0):
        """
        Computed velocity limit based on the turning radius required to reach goal.
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0
        dist_projected = lin_err * np.sin(heading_diff)
        return w_max * max(dist_projected - dead_zone, 0.0)

    def _compute_error_pose(self):
        """
        Updates error based on robot localization
        """
        xyt_loc_new = self.robot.get_estimator_pose()

        xyt_err = transform_global_to_base(self.xyt_goal, xyt_loc_new)
        if not self.track_yaw:
            xyt_err[2] = 0.0

        return xyt_err

    def _run(self):
        rate = rospy.Rate(self.hz)

        while True:
            v_cmd = w_cmd = 0
            xyt_err = self._compute_error_pose()

            lin_err_abs = np.linalg.norm(xyt_err[0:2])
            ang_err = xyt_err[2]

            # Go to goal XY position if not there yet
            if lin_err_abs > self.lin_error_tol:
                heading_err = np.arctan2(xyt_err[1], xyt_err[0])
                heading_err_abs = abs(heading_err)

                # Compute linear velocity
                v_raw = self._velocity_feedback_control(
                    lin_err_abs, ACC_LIN, self.v_max, tol=self.lin_error_tol
                )
                k_proj = self._projection_velocity_multiplier(
                    heading_err_abs, tol=self.ang_error_tol
                )
                v_limit = self._turn_rate_limit(
                    lin_err_abs,
                    heading_err_abs,
                    self.w_max / 2.0,
                    dead_zone=2.0 * self.lin_error_tol,
                )
                v_cmd = min(k_proj * v_raw, v_limit)

                # Compute angular velocity
                w_cmd = self._velocity_feedback_control(
                    heading_err, ACC_ANG, self.w_max, tol=self.ang_error_tol / 2.0, use_acc=False
                )

            # Rotate to correct yaw if yaw tracking is on and XY position is at goal
            elif abs(ang_err) > self.ang_error_tol:
                # Compute angular velocity
                w_cmd = self._velocity_feedback_control(
                    ang_err, ACC_ANG, self.w_max, tol=self.ang_error_tol, use_acc=False
                )

            # Command robot
            with self.control_lock:
                self.robot.set_velocity(v_cmd, w_cmd)

            # Spin
            rate.sleep()

    def check_at_goal(self) -> bool:
        xyt_err = self._compute_error_pose()

        xy_fulfilled = np.linalg.norm(xyt_err[0:2]) <= self.lin_error_tol

        t_fulfilled = True
        if self.track_yaw:
            t_fulfilled = abs(xyt_err[2]) <= self.ang_error_tol

        return xy_fulfilled and t_fulfilled

    def set_goal(
        self,
        xyt_position: List[float],
    ):
        self.xyt_goal = xyt_position

    def enable_yaw_tracking(self, value: bool = True):
        self.track_yaw = value

    def start(self):
        if self.loop_thr is None:
            self.loop_thr = threading.Thread(target=self._run)
            self.loop_thr.start()
        self.active = True

    def pause(self):
        self.active = False
        with self.control_lock:
            self.robot.set_velocity(0.0, 0.0)
