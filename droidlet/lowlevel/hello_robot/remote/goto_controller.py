import time
import threading

import numpy as np

V_MAX = 0.15  # base.params["motion"]["default"]["vel_m"]
W_MAX = 0.9  # 2 * (vel_m_max - vel_m_default) / wheel_separation_m
ACC_LIN = 0.2  # base.params["motion"]["default"]["accel_m"]
ACC_ANG = 1.2  # 2 * (accel_m_max - accel_m_default) / wheel_separation_m

DEFAULT_LIN_TOL = 0.01
DEFAULT_ANG_TOL = 0.05


class GotoVelocityController:
    def __init__(self, robot, hz):
        self.robot = robot
        self.dt = 1.0 / hz

        # Params
        self.a_lin = ACC_LIN
        self.a_ang = ACC_ANG

        self.v_max = V_MAX
        self.w_max = W_MAX

        self.lin_error_tol = DEFAULT_LIN_TOL
        self.ang_error_tol = DEFAULT_ANG_TOL

        # Initialize
        self.xyt_err = np.zeros(3)
        self.track_yaw = True
        self.loop_thr = None

    def _trapezoidal_velocity_multiplier(self, x_err, a):
        """
        Computes velocity multiplier based on distance from target.
        Maintains a trapezoidal velocity profile.
        Used for both linear and angular motion.
        """
        t = np.sqrt(2.0 * x_err / a)  # x_err = (1/2) * a * t^2
        return min(a * t, 1.0)

    def _projection_velocity_multiplier(self, theta_err):
        """
        Compute velocity muliplier based on yaw (faster if facing towards target).
        Used to control linear motion.
        """
        return 1.0 - np.sin(min(theta_err, np.pi / 2.0))

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

        self.xyt_err[0] = ct * x_err_f0 + st * y_err_f0
        self.xyt_err[1] = -st * x_err_f0 + ct * y_err_f0
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
                heading_err = np.arctan(self.xyt_err[1] / self.xyt_err[0])
                heading_err_abs = abs(heading_err)

                # Compute linear velocity
                k_t = self._trapezoidal_velocity_multiplier(lin_err_abs, self.a_lin)
                k_p = self._projection_velocity_multiplier(heading_err_abs)
                v_cmd = k_t * k_p * self.v_max

                # Compute angular velocity
                k_t_ang = self._trapezoidal_velocity_multiplier(heading_err_abs, self.a_ang)
                w_cmd = np.sign(heading_err) * k_t_ang * self.w_max

            # Rotate to correct yaw if yaw tracking is on and XY position is at goal
            elif self.track_yaw and ang_err_abs > self.ang_error_tol:
                # Compute angular velocity
                k_t_ang = self._trapezoidal_velocity_multiplier(ang_err_abs, self.a_ang)
                w_cmd = np.sign(ang_err) * k_t_ang * self.w_max

            # Command robot
            self.robot.set_velocity(v_cmd, w_cmd)

            # Update odometry prediction
            self._integrate_error(v_cmd, w_cmd)

            # Spin
            t_target += self.dt
            t_sleep = max(t_target - time.time(), 0.0)
            time.sleep(t_sleep)

    def check_at_goal(self):
        xy_fulfilled = np.linalg.norm(self.xyt_err[0:2]) <= self.lin_error_tol

        t_fulfilled = True
        if self.track_yaw:
            t_fulfilled = abs(self.xyt_err[2]) <= self.ang_error_tol

        return xy_fulfilled and t_fulfilled

    def set_goal(self, xyt_position, x_tol=None, r_tol=None):
        self.xyt_err = xyt_position

    def enable_yaw_tracking(self, value: bool):
        self.track_yaw = value

    def start(self):
        self.loop_thr = threading.thread(target=self._run)
        self.loop_thr.start()
