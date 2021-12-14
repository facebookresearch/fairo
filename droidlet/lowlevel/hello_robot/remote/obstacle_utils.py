import open3d as o3d
import numpy as np
import copy

def get_relative_bbox(base_position, region_shape, relative_position):
    bbox = o3d.geometry.OrientedBoundingBox(
        center=[0.0, 0.0, 0.0],
        R=np.identity((3)),
        extent=region_shape
    )
    bbox.translate(relative_position, relative=False)
    if base_position[2] != 0:
        rotz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, base_position[2]])
        bbox.rotate(rotz, center=[0.0, 0.0, 0.0])
    bbox.translate([base_position[0], base_position[1], 0.], relative=True)
    return bbox

def get_relative_points(pcd, base_position, region_shape,
                        relative_position):
    bbox = get_relative_bbox(base_position, region_shape,
                             relative_position)
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd, bbox

def get_points_in_front(pcd, base_position,
                        min_dist=0.3, max_dist=1.0,
                        robot_width=0.4, height=1.0):
    return get_relative_points(pcd, base_position,
                               [robot_width, max_dist - min_dist, height],
                               [min_dist, 0.0, 0.0])


def get_ground_plane(scan, distance_threshold=0.06, ransac_n=3, num_iterations=100, return_ground=True):
    num_points = np.asarray(scan.points).shape[0]
    if num_points < ransac_n:
        if return_ground:
            return o3d.geometry.PointCloud(), scan
        else:
            return scan
    pcd = copy.deepcopy(scan)

    ground_model, ground_indexes = scan.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
    ground_indexes = np.array(ground_indexes)

    rest = pcd.select_by_index(ground_indexes, invert=True)
    if return_ground:
        ground = pcd.select_by_index(ground_indexes)
        return ground, rest
    else:
        return rest

def is_obstacle(pcd, base_pos, pix_threshold=100, fastmath=False):
    print("num points", np.asarray(pcd.points).shape)
    crop, bbox = get_points_in_front(pcd, base_pos)
    print("num cropped", np.asarray(crop.points).shape)
    if fastmath:
        pass
    else:
        num_cropped_points = np.asarray(crop.points).shape[0]
        if num_cropped_points < pix_threshold:
            raise RuntimeError("not able to see directly in front of robot, tilt the camera further down")
        rest = get_ground_plane(crop, return_ground=False)
        rest = np.asarray(rest.points)
        print(rest.shape)
        return rest.shape[0] > 100

def get_o3d_pointcloud(points, colors):
    points, colors = points.reshape(-1, 3), colors.reshape(-1, 3)
    colors = colors / 255.
    opcd = o3d.geometry.PointCloud()
    opcd.points = o3d.utility.Vector3dVector(points)
    opcd.colors = o3d.utility.Vector3dVector(colors)
    return opcd
       
