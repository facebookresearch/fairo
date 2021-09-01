import cv2
import pytorch3d
import math

import os
import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from .pose_initializer import PoseInitializer

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class ManualInitializer(PoseInitializer):
    def __init__(self, model_configs, intrinsics, unit_in_meters):
        self.model_configs = model_configs
        self.unit_in_meters = unit_in_meters
        self.intrinsics = intrinsics

        self.f = torch.tensor(
            (self.intrinsics.fu, self.intrinsics.fv), dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # dim = (1, 2)
        self.p = torch.tensor(
            (self.intrinsics.ppu, self.intrinsics.ppv),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(
            0
        )  # dim = (1, 2)
        self.img_size = (self.intrinsics.width, self.intrinsics.height)
        print(self.img_size)
        self.meshs = {}
        for model_config, unit_in_meter in zip(self.model_configs, self.unit_in_meters):
            self.meshs[model_config.name] = {
                "mesh": load_objs_as_meshes(
                    [os.path.join(model_config.path, model_config.model_filename)],
                    device=device,
                ).scale_verts(unit_in_meter),
                "config": model_config,
            }

    def get_pose(self, image):
        pose = {}
        for mesh_name, unit_in_meter in zip(self.meshs.keys(), self.unit_in_meters):
            mesh = self.meshs[mesh_name]["mesh"]

            def on_change(value):
                img_copy = image.copy()

                x = (cv2.getTrackbarPos("x", "image") - 1000) / 1000
                y = (cv2.getTrackbarPos("y", "image") - 1000) / 1000
                z = cv2.getTrackbarPos("z", "image") / 1000
                rx = cv2.getTrackbarPos("rx", "image")
                ry = cv2.getTrackbarPos("ry", "image")
                rz = cv2.getTrackbarPos("rz", "image")

                T = torch.tensor([[x, y, z]], dtype=torch.float32, device=device)
                R = Rotation.from_euler("zyx", [rz, ry, rx], degrees=True).as_matrix()

                renderR = torch.from_numpy(R.T.reshape((1, 3, 3))).to(device)

                cameras = PerspectiveCameras(
                    R=renderR,
                    T=T,
                    focal_length=-self.f,
                    principal_point=self.p,
                    image_size=(self.img_size,),
                    device=device,
                )

                raster_settings = RasterizationSettings(
                    image_size=(self.intrinsics.height, self.intrinsics.width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    ),
                    shader=SoftPhongShader(device=device, cameras=cameras,),
                )
                overlay = renderer(mesh)[0, ..., :3].cpu().numpy()[:, :, ::-1]
                render_img = overlay * 0.7 + img_copy / 255 * 0.3
                cv2.imshow(windowName, render_img)

                store_and_exit = cv2.getTrackbarPos(
                    "0 : Manual Match \n1 : Store and Exit", "image"
                )
                if store_and_exit:
                    cv2.destroyAllWindows()
                    pose[mesh_name] = {"translation": T.cpu().numpy(), "rotation": R}

            T = torch.tensor([[0, 0, 0.5]], dtype=torch.float32, device=device)
            R = Rotation.from_euler("zyx", [0, 0, 0], degrees=True).as_matrix()
            renderR = torch.from_numpy(R.T.reshape((1, 3, 3))).to(device)

            cameras = PerspectiveCameras(
                R=renderR,
                T=T,
                focal_length=-self.f,
                principal_point=self.p,
                image_size=(self.img_size,),
                device=device,
            )

            raster_settings = RasterizationSettings(
                image_size=(self.intrinsics.height, self.intrinsics.width),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ),
                shader=SoftPhongShader(device=device, cameras=cameras,),
            )

            windowName = "image"
            overlay = (renderer(mesh)[0, ..., :3].cpu().numpy())[:, :, ::-1]
            img_copy = image.copy()

            cv2.imshow(windowName, overlay * 0.7 + img_copy / 255 * 0.3)
            cv2.createTrackbar("x", windowName, 1000, 2000, on_change)
            cv2.createTrackbar("y", windowName, 1000, 2000, on_change)
            cv2.createTrackbar("z", windowName, 500, 1000, on_change)
            cv2.createTrackbar("rx", windowName, 0, 360, on_change)
            cv2.createTrackbar("ry", windowName, 0, 360, on_change)
            cv2.createTrackbar("rz", windowName, 0, 360, on_change)
            cv2.createTrackbar(
                "0 : Manual Match \n1 : Store and Exit", windowName, 0, 1, on_change
            )

            cv2.waitKey(0)

        return pose


# mesh_path = "/home/fair-pitt/RBGT_PY/RBOT_dataset/squirrel_small.obj"
# unit_in_meter = 0.001
# f = torch.tensor((650.048, 647.183), dtype=torch.float32, device=device).unsqueeze(0) # dim = (1, 2)
# p = torch.tensor((324.328, 257.323), dtype=torch.float32, device=device).unsqueeze(0) # dim = (1, 2)
# img_size = (640, 512)
# mesh = load_objs_as_meshes([mesh_path], device=device)
# mesh = mesh.scale_verts(unit_in_meter)

# img = cv2.imread('/home/fair-pitt/RBGT_PY/RBOT_dataset/ape/frames/d_occlusion0500.png')

# T = torch.tensor([[0, 0, 0.5]], dtype=torch.float32, device=device)
# R = Rotation.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()

# renderR = torch.from_numpy(R.T.reshape((1, 3, 3))).to(device)

# cameras = PerspectiveCameras(R=renderR, T=T, focal_length=-f, principal_point=p, image_size=(img_size,), device=device)

# raster_settings = RasterizationSettings(
#     image_size=(512, 640),
#     blur_radius=0.0,
#     faces_per_pixel=1,
# )
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras,
#         raster_settings=raster_settings
#     ),
#     shader=SoftPhongShader(
#         device=device,
#         cameras=cameras,
#     )
# )

# windowName = 'image'
# overlay = (renderer(mesh)[0, ..., :3].cpu().numpy())[:, :, ::-1]
# img_copy = img.copy()

# def on_change(value):
#     img_copy = img.copy()

#     x = (cv2.getTrackbarPos('x','image') - 1000) / 1000
#     y = (cv2.getTrackbarPos('y','image') - 1000) / 1000
#     z = cv2.getTrackbarPos('z','image') / 1000
#     rx = cv2.getTrackbarPos('rx','image')
#     ry = cv2.getTrackbarPos('ry','image')
#     rz = cv2.getTrackbarPos('rz','image')

#     T = torch.tensor([[x, y, z]], dtype=torch.float32, device=device)
#     R = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True).as_matrix()

#     renderR = torch.from_numpy(R.T.reshape((1, 3, 3))).to(device)

#     cameras = PerspectiveCameras(R=renderR, T=T, focal_length=-f, principal_point=p, image_size=(img_size,), device=device)

#     raster_settings = RasterizationSettings(
#         image_size=(512, 640),
#         blur_radius=0.0,
#         faces_per_pixel=1,
#     )
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras,
#             raster_settings=raster_settings
#         ),
#         shader=SoftPhongShader(
#             device=device,
#             cameras=cameras,
#         )
#     )
#     overlay = (renderer(mesh)[0, ..., :3].cpu().numpy()[:, :, ::-1])
#     render_img = (overlay * 0.7 + img_copy / 255 * 0.3)
#     cv2.imshow(windowName, render_img)

#     store_and_exit = cv2.getTrackbarPos('0 : Manual Match \n1 : Store and Exit','image')
#     if store_and_exit:
#         print(T, R)
#         cv2.destroyAllWindows()

# cv2.imshow(windowName, overlay*0.7 + img_copy / 255 * 0.3)
# cv2.createTrackbar('x', windowName, 1000, 2000, on_change)
# cv2.createTrackbar('y', windowName, 1000, 2000, on_change)
# cv2.createTrackbar('z', windowName, 500, 1000, on_change)
# cv2.createTrackbar('rx', windowName, 0, 360, on_change)
# cv2.createTrackbar('ry', windowName, 0, 360, on_change)
# cv2.createTrackbar('rz', windowName, 0, 360, on_change)
# cv2.createTrackbar('0 : Manual Match \n1 : Store and Exit', windowName, 0, 1, on_change)

# cv2.waitKey(0)
