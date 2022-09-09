from typing import List, Optional
import time
import threading

import numpy as np

V_MAX = 0.15  # base.params["motion"]["default"]["vel_m"]
W_MAX = 0.9  # 2 * (vel_m_max - vel_m_default) / wheel_separation_m
DEFAULT_LIN_TOL = 0.005
DEFAULT_ANG_TOL = 0.025


class GotoVelocityController:
    def __init__(self, robot, hz):
        self.robot = robot
        self.dt = 1.0 / hz

        # Params
        self.v_max = V_MAX
        self.w_max = W_MAX
        self.lin_error_tol = DEFAULT_LIN_TOL
        self.ang_error_tol = DEFAULT_ANG_TOL

        # Initialize
        self.xyt_err = np.zeros(3)
        self.track_yaw = True
        self.loop_thr = None

    @staticmethod
    def _error_velocity_multiplier(x_err, tol=0.0):
        """
        Computes velocity multiplier based on distance from target.
        Used for both linear and angular motion.
        """
        assert x_err >= 0.0
        return float(x_err - tol > 0)

    @staticmethod
    def _projection_velocity_multiplier(theta_err, tol=0.0):
        """
        Compute velocity muliplier based on yaw (faster if facing towards target).
        Used to control linear motion.
        """
        assert theta_err >= 0.0
        return 1.0 - np.sin(min(max(theta_err - tol, 0.0) * 2.0, np.pi / 2.0))

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
        self.xyt_err[2] = self.xyt_err[2] - dtheta

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
                k_t = self._error_velocity_multiplier(lin_err_abs)
                k_p = self._projection_velocity_multiplier(heading_err_abs, tol=self.ang_error_tol)
                v_cmd = k_t * k_p * self.v_max

                # Compute angular velocity
                k_t_ang = self._error_velocity_multiplier(heading_err_abs, tol=0.0)
                w_cmd = np.sign(heading_err) * k_t_ang * self.w_max

            # Rotate to correct yaw if yaw tracking is on and XY position is at goal
            elif self.track_yaw and ang_err_abs > self.ang_error_tol:
                # Compute angular velocity
                k_t_ang = self._error_velocity_multiplier(ang_err_abs)
                w_cmd = np.sign(ang_err) * k_t_ang * self.w_max

            # Command robot
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
        x_tol: Optional[float] = None,
        r_tol: Optional[float] = None,
    ):
        self.xyt_err = xyt_position
        self.lin_error_tol = x_tol or DEFAULT_LIN_TOL
        self.ang_error_tol = r_tol or DEFAULT_ANG_TOL

    def enable_yaw_tracking(self, value: bool):
        self.track_yaw = value

    def start(self):
        self.loop_thr = threading.Thread(target=self._run)
        self.loop_thr.start()
