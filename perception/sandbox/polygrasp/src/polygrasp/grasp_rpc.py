import logging
from typing import List
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import a0

import graspnetAPI
from . import wait_until_a0_server_ready, start_a0_server_heartbeat
from polygrasp import serdes
from polygrasp.serdes import polygrasp_msgs

import signal


log = logging.getLogger(__name__)
topic_key = "grasp_server"
grasp_topic_key = f"{topic_key}/grasp"
collision_topic_key = f"{topic_key}/collision"


def save_img(img, name):
    f = plt.figure()
    plt.imshow(img)
    f.savefig(f"{name}.png")
    plt.close(f)


class GraspServer:
    def _get_grasps(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        raise NotImplementedError

    def _get_collisions(
        self, grasps: graspnetAPI.GraspGroup, scene_pcd: o3d.geometry.PointCloud
    ) -> graspnetAPI.GraspGroup:
        raise NotImplementedError

    def start(self):
        log.info(f"Starting grasp server...")

        def grasp_onrequest(req):
            log.info(f"Got request; computing grasp group...")

            payload = req.pkt.payload
            pcd = serdes.capnp_to_pcd(payload)
            grasp_group = self._get_grasps(pcd)

            log.info(f"Done. Replying with serialized grasp group...")
            req.reply(serdes.grasp_group_to_capnp(grasp_group).to_bytes())

        def collision_onrequest(req):
            """
            Calls the collision detector from graspnet-baseline's server.
            """
            log.info(f"Got request; computing collisions...")

            payload = req.pkt.payload
            with polygrasp_msgs.CollisionRequest.from_bytes(payload) as msg:
                grasp_group = serdes.bytes_to_grasp_group(msg.grasps)
                scene_pcd = serdes.capnp_to_pcd(msg.pcd)

            filtered_grasp_group = self._get_collisions(grasp_group, scene_pcd)
            log.info(f"Done. Replying with serialized filtered grasps...")
            req.reply(serdes.grasp_group_to_bytes(filtered_grasp_group))

        self.grasp_server = a0.RpcServer(grasp_topic_key, grasp_onrequest, None)
        start_a0_server_heartbeat(grasp_topic_key)
        self.collision_server = a0.RpcServer(
            collision_topic_key, collision_onrequest, None
        )
        start_a0_server_heartbeat(collision_topic_key)

        signal.pause()


class GraspClient:
    def __init__(self, view_json_path):
        wait_until_a0_server_ready(grasp_topic_key)
        wait_until_a0_server_ready(collision_topic_key)

        self.grasp_client = a0.RpcClient(grasp_topic_key)
        self.collision_client = a0.RpcClient(collision_topic_key)
        self.view_json_path = view_json_path

    def downsample_pcd(
        self, pcd: o3d.geometry.PointCloud, max_num_bits=8 * 1024 * 1024
    ):
        # a0 default max msg size 16MB; make sure every msg < 1/2 of max
        i = 1
        while True:
            downsampled_pcd = pcd.uniform_down_sample(i)
            bits = serdes.pcd_to_capnp(downsampled_pcd).to_bytes()
            if len(bits) > max_num_bits:
                log.warning(f"Downsampling pointcloud...")
                i += 1
            else:
                break
        if i > 1:
            log.warning(f"Downsampled to every {i}th point.")

        return bits

    def get_grasps(self, pcd: o3d.geometry.PointCloud) -> graspnetAPI.GraspGroup:
        bits = self.downsample_pcd(pcd)
        result_bits = self.grasp_client.send_blocking(bits).payload
        return serdes.capnp_to_grasp_group(result_bits)

    def get_collision(
        self, grasps: graspnetAPI.GraspGroup, scene_pcd: o3d.geometry.PointCloud
    ):
        request = polygrasp_msgs.CollisionRequest()
        request.pcd = self.downsample_pcd(scene_pcd)
        request.grasps = serdes.grasp_group_to_bytes(grasps)

        bits = request.to_bytes()
        result_bits = self.collision_client.send_blocking(bits).payload

        return serdes.bytes_to_grasp_group(result_bits)

    def visualize(self, scene_pcd, render=False, save_view=False):
        """Render a scene's pointcloud and return the Open3d Visualizer."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(scene_pcd, reset_bounding_box=True)

        if render:
            """Render the window. You can rotate it & save the view."""
            # Actually render the window:
            log.info(f"Rendering scene in Open3D")
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            if save_view:
                log.info(f"Saving new view to {self.view_json_path}")
                # Save the view
                o3d.io.write_pinhole_camera_parameters(self.view_json_path, param)

        param = o3d.io.read_pinhole_camera_parameters(self.view_json_path)
        vis.get_view_control().convert_from_pinhole_camera_parameters(param)

        return vis

    def visualize_grasp(
        self,
        scene_pcd: o3d.geometry.PointCloud,
        grasp_group: graspnetAPI.GraspGroup,
        n=5,
        render=False,
        save_view=False,
        name="scene",
    ) -> None:
        """Visualize grasps upon a scene's pointcloud."""
        grasp_o3d_geometries = grasp_group.to_open3d_geometry_list()
        grasp_pointclouds = [
            grasp_o3d_geometry.sample_points_uniformly(number_of_points=5000)
            for grasp_o3d_geometry in grasp_o3d_geometries
        ]
        vis = self.visualize(scene_pcd=scene_pcd, render=render, save_view=save_view)

        # Save scene
        grasp_image = np.array(vis.capture_screen_float_buffer(do_render=True))
        save_img(grasp_image, name)

        n = min(n, len(grasp_o3d_geometries))
        log.info(f"Visualizing top {n} grasps in Open3D...")

        # Save scene with each top grasp individually
        for i, grasp_pointcloud in enumerate(grasp_pointclouds[:n]):
            vis.add_geometry(grasp_pointcloud, reset_bounding_box=False)
            grasp_image = np.array(vis.capture_screen_float_buffer(do_render=True))
            save_img(grasp_image, f"{name}_with_grasp_{i + 1}")
            vis.remove_geometry(grasp_pointcloud, reset_bounding_box=False)

        # Save scene with all grasps
        for grasp_pointcloud in grasp_pointclouds[:n]:
            vis.add_geometry(grasp_pointcloud, reset_bounding_box=False)
        grasp_image = np.array(vis.capture_screen_float_buffer(do_render=True))
        save_img(grasp_image, f"{name}_with_grasps")

        return vis

    def get_obj_grasps(
        self,
        obj_pcds: List[o3d.geometry.PointCloud],
        scene_pcd: o3d.geometry.PointCloud,
    ):
        """
        Get grasps for each object pointcloud, then filter by
        checking collisions against the scene pointcloud.
        """
        for obj_i, obj_pcd in enumerate(obj_pcds):
            print(f"Getting obj {obj_i} grasp...")
            grasp_group = self.get_grasps(obj_pcd)
            filtered_grasp_group = self.get_collision(grasp_group, scene_pcd)
            if len(filtered_grasp_group) < len(grasp_group):
                print(
                    "Filtered"
                    f" {len(grasp_group) - len(filtered_grasp_group)}/{len(grasp_group)} grasps"
                    " due to collision."
                )
            if len(filtered_grasp_group) > 0:
                return obj_i, filtered_grasp_group
        raise Exception(
            "Unable to find any grasps after filtering, for any of the"
            f" {len(obj_pcds)} objects"
        )
