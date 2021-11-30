import os
import sys
import threading
import json
import copy
import time
import shutil
import base64

import a0
import torch
import pybullet as p


class MoveitInterface:
    def __init__(self):
        self.moveit_client = a0.RpcClient("panda_planner")

        # Threading
        self.cv = threading.Condition()
        self.return_flag = False
        self.result_buffer = None

    @staticmethod
    def _unpack(pkt):
        reply = json.loads(pkt.payload.decode())
        if reply["err"] != "success":
            print(reply["result"], flush=True, file=sys.stderr)
            raise
        return reply["result"]

    def _a0_callback(self, pkt):
        with self.cv:
            self.result_buffer = self._unpack(pkt)
            self.return_flag = True
            self.cv.notify()

    def _query_moveit(self, request):
        self.return_flag = False
        self.moveit_client.send(request, self._a0_callback)
        with self.cv:
            self.cv.wait_for(lambda: self.return_flag)

        return copy.deepcopy(self.result_buffer)

    @staticmethod
    def interpolate_trajectory(trajectory, hz):
        new_trajectory = []

        dt = 1.0 / hz

        idx = 0
        t = 0.0
        while True:
            # Find interval
            while idx < len(trajectory) - 1 and t > trajectory[idx + 1]["secs_from_start"]:
                idx += 1
            if idx >= len(trajectory) - 1:
                break

            # Interpolate
            t0 = trajectory[idx]["secs_from_start"]
            t1 = trajectory[idx + 1]["secs_from_start"]
            r = (t - t0) / (t1 - t0)

            new_point = {"secs_from_start": t}
            for field in ["joint_positions", "joint_velocities", "joint_accelerations"]:
                new_point[field] = (1 - r) * trajectory[idx][field] + r * trajectory[idx + 1][
                    field
                ]
            new_trajectory.append(new_point)

            # Increment time
            t += dt

        return new_trajectory

    @staticmethod
    def _convert_pose(pos, quat):
        return {
            "position": {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            },
            "orientation": {
                "x": float(quat[0]),
                "y": float(quat[1]),
                "z": float(quat[2]),
                "w": float(quat[3]),
            },
        }

    def add_mesh(self, name, mesh_pos, mesh_quat, filename):
        # Create tmpfile
        abs_filename = os.path.abspath(filename)

        basename = os.path.basename(abs_filename)
        basename_ext = f".{basename.split('.')[-1]}"
        basename_name = basename.strip(basename_ext)

        dirname = os.path.dirname(abs_filename)
        dirname_encoding = base64.b64encode(bytes(dirname, "utf-8")).decode()

        filename_target = f"/tmp/mesh/{basename_name}_{dirname_encoding}{basename_ext}"
        shutil.copyfile(filename, filename_target)

        # Convert pose
        mesh_pose = self._convert_pose(mesh_pos, mesh_quat)

        # Send request
        self._query_moveit(f"scene.add_mesh('{name}', {mesh_pose}, '{filename_target}')")

    def plan(self, current_joint_pos, target_pos, target_quat, ee_link, time_to_go, hz=None):
        # Set current state
        state = self._query_moveit(f"move_group.get_current_state()")
        for i in range(7):
            state["joint_state"]["position"][i] = float(current_joint_pos[i])
        self._query_moveit(f"move_group.set_start_state({state})")
        time.sleep(1)

        # Set target
        target_pose = self._convert_pose(target_pos, target_quat)
        self._query_moveit(f"move_group.set_pose_target({target_pose})")

        # Print planning parameters
        """
        print("===========================")
        print(json.dumps(self._query_moveit(f"move_group.construct_motion_plan_request()"), indent="  "))
        print("===========================")
        """

        # Plan
        success, traj_raw, planning_time, error_code = self._query_moveit(f"move_group.plan()")

        # Extract trajectory
        if success:
            trajectory = [
                {
                    "secs_from_start": point["time_from_start"]["secs"]
                    + 1e-9 * point["time_from_start"]["nsecs"],
                    "joint_positions": torch.Tensor(point["positions"]),
                    "joint_velocities": torch.Tensor(point["velocities"]),
                    "joint_accelerations": torch.Tensor(point["accelerations"]),
                }
                for point in traj_raw["joint_trajectory"]["points"]
            ]

            # Interpolate trajectory
            if hz is not None:
                trajectory = self.interpolate_trajectory(trajectory, hz)
        else:
            trajectory = None

        # Extract info
        planning_info = {
            "success": success,
            "planning_time": planning_time,
            "error_code": error_code,
        }

        return trajectory, planning_info
