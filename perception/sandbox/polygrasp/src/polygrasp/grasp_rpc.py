import logging
import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt

import a0

import graspnetAPI
from polygrasp import serdes


log = logging.getLogger(__name__)
topic_key = "grasp_server"


class GraspServer:
    def _get_grasps(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        raise NotImplementedError

    def start(self):
        log.info(f"Starting grasp server...")

        def onrequest(req):
            log.info(f"Got request; computing grasp group...")

            payload = req.pkt.payload
            pcd = serdes.capnp_to_pcd(payload)
            grasp_group = self._get_grasps(pcd)

            log.info(f"Done. Replying with serialized grasp group...")
            req.reply(serdes.grasp_group_to_capnp(grasp_group).to_bytes())

        server = a0.RpcServer(topic_key, onrequest, None)
        while True:
            pass


class GraspClient:
    def __init__(self, view_json_path):
        self.client = a0.RpcClient(topic_key)
        self.view_json_path = view_json_path

    def get_grasps(self, pcd: o3d.geometry.PointCloud) -> graspnetAPI.GraspGroup:
        state = []

        def onreply(pkt):
            state.append(pkt.payload)

        i = 1
        while True:
            downsampled_pcd = pcd.uniform_down_sample(i)
            bits = serdes.pcd_to_capnp(downsampled_pcd).to_bytes()
            if (
                len(bits) > 8 * 1024 * 1024
            ):  # a0 default max msg size 16MB; make sure every msg < 1/2 of max
                log.warning(f"Downsampling pointcloud...")
                i += 1
            else:
                break
        if i > 1:
            log.warning(f"Downsampled to every {i}th point.")
        self.client.send(bits, onreply)

        while not state:
            pass
        return serdes.capnp_to_grasp_group(state.pop())
    
    def visualize(self, scene_pcd, plot=False, render=False, save_view=False):
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
    
    def get_rgbd(self, vis):
        rgb = np.array(vis.capture_screen_float_buffer(do_render=True))
        d = np.array(vis.capture_depth_float_buffer(do_render=True))
        intrinsics = vis.get_view_control().convert_to_pinhole_camera_parameters()

        import pdb; pdb.set_trace()


    def visualize_grasp(
        self,
        scene_pcd: o3d.geometry.PointCloud,
        grasp_group: graspnetAPI.GraspGroup,
        n=5,
        plot=False,
        render=False,
        save_view=False,
    ) -> None:
        vis = self.visualize(scene_pcd=scene_pcd, plot=plot, render=render, save_view=save_view)

        o3d_geometries = grasp_group.to_open3d_geometry_list()
        n = min(n, len(o3d_geometries))
        log.info(f"Visualizing top {n} grasps in Open3D...")

        grasp_images = []
        for i in range(n):
            scene_points = o3d_geometries[i].sample_points_uniformly(number_of_points=5000)
            vis.add_geometry(scene_points, reset_bounding_box=False)
            grasp_image = np.array(vis.capture_screen_float_buffer(do_render=True))
            grasp_images.append(grasp_image)
            vis.remove_geometry(scene_points, reset_bounding_box=False)

        if plot:
            log.info("Plotting with matplotlib...")
            f, axarr = plt.subplots(1, n, figsize=(n * 4.5, 3))
            for i in range(n):
                axarr[i].imshow(grasp_images[i], interpolation="nearest", aspect="auto")
                axarr[i].axis("off")
                axarr[i].set_title(f"Grasp pose top {i + 1}/{n}")
            f.show()
        
        return vis
