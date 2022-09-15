from typing import List, Optional
import time
import threading

import numpy as np

V_MAX_DEFAULT = 0.15  # base.params["motion"]["default"]["vel_m"]
W_MAX_DEFAULT = 0.45  # (vel_m_max - vel_m_default) / wheel_separation_m


class GotoVelocityController:
    def __init__(
        self,
        robot,
        hz: float,
        v_max: Optional[float] = None,
        w_max: Optional[float] = None,
    ):
        self.robot = robot
        self.dt = 1.0 / hz

        # Params
        self.v_max = v_max or V_MAX_DEFAULT
        self.w_max = w_max or W_MAX_DEFAULT
        self.lin_error_tol = self.v_max / hz
        self.ang_error_tol = self.w_max / hz

        # Initialize
        self.loop_thr = None
        self.control_lock = threading.Lock()
        self.active = False

        self.xyt_err = np.zeros(3)
        self.track_yaw = True

    @staticmethod
    def _error_velocity_multiplier(x_err, tol=0.0):
        """
        Computes velocity multiplier based on distance from target.
        Used for both linear and angular motion.

        Current implementation: Simple thresholding
        Output = 1 if linear error is larger than the tolerance, 0 otherwise.
        """
        assert x_err >= 0.0
        return float(x_err - tol > 0)

    @staticmethod
    def _projection_velocity_multiplier(theta_err, tol=0.0):
        """
        Compute velocity muliplier based on yaw (faster if facing towards target).
        Used to control linear motion.

        Current implementation:
        Output = 1 when facing target, gradually decreases to 0 when angle to target is pi/3.
        """
        assert theta_err >= 0.0
        return 1.0 - np.sin(min(max(theta_err - tol, 0.0) * 2.0, np.pi / 3.0))

    @staticmethod
    def _turn_rate_limit(w_max, lin_err, heading_err):
        """
        Computed velocity limit based on the turning radius required to reach goal.
        """
        assert lin_err >= 0.0
        assert heading_err >= 0.0
        return w_max * lin_err / np.sin(heading_err) + 1e-5 / 2.0

    def _integrate_state(self, v, w):
        """
        Predict error in the next timestep with current commanded velocity
        """
        dx = v * self.dt
        dtheta = w * self.dt

        x_err_f0 = self.xyt_err[0] - dx * np.cos(dtheta / 2.0)
        y_err_f0 = self.xyt_err[1] - dx * np.sin(dtheta / 2.0)
        ct = np.cos(-dtheta)
        st = np.sin(-dtheta)

        self.xyt_err[0] = ct * x_err_f0 - st * y_err_f0
        self.xyt_err[1] = st * x_err_f0 + ct * y_err_f0
        self.xyt_err[2] = self.xyt_err[2] - dtheta if self.track_yaw else 0.0

    def _run(self):
        t_target = time.time()

        while True:
            v_cmd = w_cmd = 0

            lin_err_abs = np.linalg.norm(self.xyt_err[0:2])
            ang_err = self.xyt_err[2]
            ang_err_abs = abs(ang_err)

            # Go to goal XY position if not there yet
            if lin_err_abs > self.lin_error_tol:
                heading_err = np.arctan2(self.xyt_err[1], self.xyt_err[0])
                heading_err_abs = abs(heading_err)

                # Compute linear velocity
                k_t = self._error_velocity_multiplier(lin_err_abs, tol=self.lin_error_tol)
                k_p = self._projection_velocity_multiplier(heading_err_abs, tol=self.ang_error_tol)
                v_limit = self._turn_rate_limit(self.w_max, lin_err_abs, heading_err_abs)
                v_cmd = min(k_t * k_p * self.v_max, v_limit)

                # Compute angular velocity
                k_t_ang = self._error_velocity_multiplier(heading_err_abs, tol=self.ang_error_tol)
                w_cmd = np.sign(heading_err) * k_t_ang * self.w_max

            # Rotate to correct yaw if yaw tracking is on and XY position is at goal
            elif ang_err_abs > self.ang_error_tol:
                # Compute angular velocity
                k_t_ang = self._error_velocity_multiplier(ang_err_abs, tol=self.ang_error_tol)
                w_cmd = np.sign(ang_err) * k_t_ang * self.w_max

            # Command robot
            with self.control_lock:
                self.robot.set_velocity(v_cmd, w_cmd)

            # Update odometry prediction
            self._integrate_state(v_cmd, w_cmd)

            # Spin
            t_target += self.dt
            t_sleep = max(t_target - time.time(), 0.0)
            time.sleep(t_sleep)

    def check_at_goal(self) -> bool:
        xy_fulfilled = np.linalg.norm(self.xyt_err[0:2]) <= self.lin_error_tol

        t_fulfilled = True
        if self.track_yaw:
            t_fulfilled = abs(self.xyt_err[2]) <= self.ang_error_tol

        return xy_fulfilled and t_fulfilled

    def set_goal(
        self,
        xyt_position: List[float],
    ):
        self.xyt_err = xyt_position
        if not self.track_yaw:
            self.xyt_err[2] = 0.0

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
