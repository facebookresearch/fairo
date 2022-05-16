import time
import numpy as np
import math
import open3d as o3d
from droidlet.interpreter.robot.objects import AttributeDict


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return (pcd_down, pcd_fpfh)


def pointcloud_registration_global(src_pcd, dst_pcd, voxel_size=0.05):
    distance_threshold = voxel_size * 0.9
    src_down, src_fpfh = preprocess_point_cloud(src_pcd, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst_pcd, voxel_size)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=64,
            maximum_tuple_count=1000,
        ),
    )
    src_pcd.transform(result.transformation)
    return src_pcd


def pointcloud_registration_local(src_pcd, dst_pcd, initial_transform=None, voxel_size=0.05):
    if initial_transform is None:
        initial_transform = np.identity(4)
    if voxel_size is not None:
        src_down = src_pcd.voxel_down_sample(voxel_size)
        dst_down = dst_pcd.voxel_down_sample(voxel_size)
    else:
        src_down = src_pcd
        dst_down = dst_pcd
    distance_threshold = voxel_size * 0.9
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    dst_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )

    result = o3d.pipelines.registration.registration_colored_icp(
        src_down, dst_down, distance_threshold
    )
    src_pcd.transform(result.transformation)
    return src_pcd


def rgbd_registration_global(src_rgbd, dst_rgbd, initial_transform):
    pass


def rgbd_registration_local(src_rgbd, dst_rgbd, initial_transform):
    pass


class SLAM:
    def __init__(self, intrinsic, depth_ref):
        config = AttributeDict(
            {
                "name": "Default reconstruction system config",
                "device": "CPU:0",
                "depth_min": 0.1,
                "depth_max": 10.0,  # 3.0,
                "depth_scale": 1000.0,
                "odometry_distance_thr": 0.07,  # 0.07,
                "voxel_size": 0.0058,
                "trunc_voxel_multiplier": 8.0,
                "block_count": 40000,
                "est_point_count": 6000000,
                "surface_weight_thr": 3.0,
            }
        )
        self.config = config
        self.device = o3d.core.Device(config.device)
        device = self.device
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

        self.T_frame_to_model = o3d.core.Tensor(np.identity(4))
        self.model = o3d.t.pipelines.slam.Model(
            config.voxel_size, 16, config.block_count, self.T_frame_to_model, device
        )

        self.input_frame = o3d.t.pipelines.slam.Frame(
            depth_ref.rows, depth_ref.columns, intrinsic, device
        )
        self.raycast_frame = o3d.t.pipelines.slam.Frame(
            depth_ref.rows, depth_ref.columns, intrinsic, device
        )

        self.count = 0
        self.poses = []

    def update_map(self, color, depth, first=False):
        device = self.device
        T_frame_to_model = self.T_frame_to_model
        model = self.model
        input_frame = self.input_frame
        raycast_frame = self.raycast_frame
        start = time.time()
        config = self.config
        i = self.count
        self.count += 1

        depth = depth.to(device)
        color = color.to(device)

        input_frame.set_data_from_image("depth", depth)  # open3d.t.geometry.Image
        input_frame.set_data_from_image("color", color)

        if not first:
            result = model.track_frame_to_model(
                input_frame,
                raycast_frame,
                config.depth_scale,
                config.depth_max,
                config.odometry_distance_thr,
            )
            T_frame_to_model = T_frame_to_model @ result.transformation
            self.T_frame_to_model = T_frame_to_model

        self.poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(
            input_frame, config.depth_scale, config.depth_max, config.trunc_voxel_multiplier
        )
        model.synthesize_model_frame(
            raycast_frame,
            config.depth_scale,
            config.depth_min,
            config.depth_max,
            config.trunc_voxel_multiplier,
            False,
        )
        stop = time.time()
        print("{:04d} slam takes {:.4}s".format(i, stop - start))

        return model.voxel_grid, self.poses, self.config
