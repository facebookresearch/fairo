"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
import numpy as np
from droidlet.lowlevel.robot_mover_utils import transform_pose
from droidlet.shared_data_structs import RGBDepth
from droidlet.lowlevel.hello_robot.rotation import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
)
import cv2
import math
from numba import njit
from math import ceil, floor, isnan
from collections import deque, defaultdict

d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
) 

# Values for locobot in habitat. 
# TODO: generalize this for all robots
fx, fy = 256, 256
cx, cy = 256, 256

#locoboot
# intrinsic_mat = np.array([[  fx, 0., cx],
#                             [  0., fy, cy],
#                             [  0., 0., 1.]])

intrinsic_mat = np.array(
    [[604.50262451,   0.        , 312.43200684],
       [  0.        , 604.22351074, 236.35299683],
       [  0.        ,   0.        ,   1.        ]]
)

# fx, fy = 605.2880249, 605.65637207
# cx, cy = 319.11114502, 239.48382568
# intrinsic_mat = np.array([[  fx, 0., cx],
#                             [  0., fy, cy],
#                             [  0., 0., 1.]])

# rotation from pyrobot to canonical coordinates (https://github.com/facebookresearch/fairo/blob/main/agents/locobot/coordinates.MD)
# rot = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# CAMERA_HEIGHT = 0.6
# trans = np.array([0, 0, CAMERA_HEIGHT])

# my hello
# trans = np.array([ 0.03492126, -0.01123962,  1.24354383])

# rot = np.array([[ 0.84003093,  0.53898409, -0.06200145],
#  [-0.04361892, -0.04681613, -0.99795072],
#  [-0.54078223,  0.84101391, -0.01581709]])

# soumith
trans = np.array([0.02283596, 0.01864796, 1.25382417])
rot = np.array([
    [0.86391456, 0.49976488, 0.06234341],
    [0.05324502, 0.03236112, -0.99805373],
    [-0.50081594, 0.86555262, 0.00143363]
])


 # TODO: Consolidate camera intrinsics and their associated utils across locobot and habitat.
def compute_uvone(height, width):
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    img_resolution = (height, width)
    img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
    img_pixs = img_pixs.reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
    return uv_one_in_cam, intrinsic_mat, rot, trans

def uncompute_pcd(rgbd, rot_cam, trans_cam, base_state, uv_one_in_cam):
        # rgb = np.asarray(rgb).astype(np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # depth = depth.astype(np.float32)

        # # the realsense pointcloud seems to produce some spurious points
        # # really far away. So, limit the depth to 8 metres
        # thres = 8000
        # depth[depth > thres] = thres

        # depth_copy = np.copy(depth)

        # depth = depth.reshape(rgb.shape[0] * rgb.shape[1])

        # # normalize by the camera's intrinsic matrix
        # pts_in_cam = np.multiply(uv_one_in_cam, depth)
        # pts = pts_in_cam.T

        pts = rgbd.ptcloud

        translation_vector = np.array(
            [trans_cam[0] + base_state[0], trans_cam[1] + base_state[1], trans_cam[2] + 0]
        ).reshape(-1)

        pts = pts - translation_vector

        roty90 = rotation_matrix_y(90)
        rotxn90 = rotation_matrix_x(-90)
        # next, rotate and translate pts by
        # the robot pose and location
        rot_base = rotation_matrix_z(math.degrees(base_state[2]))

        rotation_matrix = rot_base @ rot_cam @ rotxn90 @ roty90
        print(f'rgbd.depth.shape {rgbd.depth.shape}, pts.shape {pts.shape}')
        pts = np.dot(pts, np.linalg.inv(rotation_matrix).T)

        pts_in_cam = np.multiply(uv_one_in_cam, pts.reshape((3, rgbd.rgb.shape[0]*rgbd.rgb.shape[1])))

        print(f'pts_in_cam.shape from uncompute_pcd {pts_in_cam.shape}')
        return pts_in_cam.T #[0].reshape(rgbd.rgb.shape[0], rgbd.rgb.shape[1])

def compute_pcd(rgb, depth, rot_cam, trans_cam, base_state, uv_one_in_cam):
        rgb = np.asarray(rgb).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = depth.astype(np.float32)

        # the realsense pointcloud seems to produce some spurious points
        # really far away. So, limit the depth to 8 metres
        thres = 8000
        depth[depth > thres] = thres

        depth_copy = np.copy(depth)

        depth = depth.reshape(rgb.shape[0] * rgb.shape[1])

        # normalize by the camera's intrinsic matrix
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        pts = pts_in_cam.T

        # Now, the points are in camera frame.
        # In camera frame
        # z is positive into the camera
        # (larger the z, more into the camera)
        # x is positive to the right
        # (larger the x, more right of the origin)
        # y is positive to the bottom
        # (larger the y, more to the bottom of the origin)
        #                                 /
        #                                /
        #                               / z-axis
        #                              /
        #                             /_____________ x-axis (640)
        #                             |
        #                             |
        #                             | y-axis (480)
        #                             |
        #                             |

        # We now need to transform this to pyrobot frame, where
        # x is into the camera, y is positive to the left,
        # z is positive upwards
        # https://pyrobot.org/docs/navigation
        #                            |    /
        #                 z-axis     |   /
        #                            |  / x-axis
        #                            | /
        #  y-axis        ____________|/
        #
        # If you hold the first configuration in your right hand, and
        # visualize the transformations needed to get to the second
        # configuration, you'll see that
        # you have to rotate 90 degrees anti-clockwise around the y axis, and then
        # 90 degrees clockwise around the x axis.
        # This results in the final configuration
        roty90 = rotation_matrix_y(90)
        rotxn90 = rotation_matrix_x(-90)
        # next, rotate and translate pts by
        # the robot pose and location
        rot_base = rotation_matrix_z(math.degrees(base_state[2]))

        rotation_matrix = rot_base @ rot_cam @ rotxn90 @ roty90
        translation_vector = np.array(
            [trans_cam[0] + base_state[0], trans_cam[1] + base_state[1], trans_cam[2] + 0]
        ).reshape(-1)

        pts = np.dot(pts, rotation_matrix.T)
        pts = pts + translation_vector

        # now rewrite the ordering of pts so that the colors (rgb_rotated)
        # match the indices of pts
        # pts = pts.reshape((rgb.shape[0], rgb.shape[1], 3))
        # pts = np.rot90(pts, k=1, axes=(1, 0))
        # pts = pts.reshape(rgb.shape[0] * rgb.shape[1], 3)

        # depth_rotated = np.rot90(depth_copy, k=1, axes=(1, 0))
        # rgb_rotated = np.rot90(rgb, k=1, axes=(1, 0))

        # return RGBDepth(rgb_rotated, depth_rotated, pts)
        return RGBDepth(rgb, depth, pts)

def convert_depth_to_pcd(depth, pose, uv_one_in_cam, rot, trans):
    # point cloud in camera frame
    depth = (depth.astype(np.float32) / 1000.0).reshape(-1)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
    # point cloud in robot base frame
    pts_in_base = pts_in_cam[:3, :].T
    pts_in_base = np.dot(pts_in_base, rot.T)
    pts_in_base = pts_in_base + trans.reshape(-1)
    # point cloud in world frame (pyrobot)
    pts_in_world = transform_pose(pts_in_base, pose)
    return pts_in_world

# @njit
def get_annot(height, width, pts_in_cur_img, src_pts_in_cur_cam, cur_pts_in_cur_cam, src_label, valid_z):
    """
    This creates the new semantic labels of the projected points in the current image frame. Each new semantic label is the 
    semantic label corresponding to pts_in_cur_img in src_label. 
    """
    annot_img = np.zeros((height, width)).astype(np.float32)
    print(f'annot_img.shape {annot_img.shape}, src_label.shape {src_label.shape}')
    for indx in range(len(pts_in_cur_img)):
        r = int(indx/width)
        c = int(indx - r*width)
        x, y, _ = pts_in_cur_img[indx]
        # We take ceil and floor combinations to fix quantization errors
        if not isnan(x) and not isnan(y) and floor(x) >= 0 and ceil(x) < width and floor(y) >=0 and ceil(y) < height and valid_z[indx]:
            cur_indx = ceil(x) + ceil(y) * width
            try:
                # if src_pts_in_cur_cam[indx][2] - cur_pts_in_cur_cam[cur_indx][2] < 0.1:
                annot_img[ceil(y)][ceil(x)] = src_label[r][c]
                annot_img[floor(y)][floor(x)] = src_label[r][c]
                annot_img[ceil(y)][floor(x)] = src_label[r][c]
                annot_img[floor(y)][ceil(x)] = src_label[r][c]
            except Exception as ex:
                print(f'caught exception {ex}')
                print(f'r c x y {r, c, x, y}')
                raise ex
    return annot_img
    def closest_non_zero(a, x, y):
        h, w = len(a), len(a[0])
        def get_neighbors(a, curx, cury):
            ns = []
            if curx > 0:
                ns.append((curx-1, cury)) # n
            if cury > 0:
                ns.append((curx, cury-1)) # w
            if cury < w-1:
                ns.append((curx, cury+1)) # e 
            if curx < h-1:
                ns.append((curx+1, cury)) # s
            if curx > 0 and cury > 0:
                ns.append((curx-1, cury-1)) #nw
            if curx > 0 and cury < w-1:
                ns.append((curx-1, cury+1)) #ne
            if curx < h-1 and cury < w-1:
                ns.append((curx+1, cury+1)) #se
            if curx < h-1 and cury > 0:
                ns.append((curx+1, cury-1)) #sw 
            return ns

        bfsq = deque([])
        visited = np.zeros_like(a)
        bfsq.append((x,y))
        pop_count = 0
        push_count = 1
        while len(bfsq) > 0:
            curx, cury = bfsq.popleft()
            pop_count += 1
            # if pop_count % 100 == 0:
                # print(f'pop_count {pop_count}')
            visited[curx][cury] = 1
            if a[curx][cury] > 0:
                return a[curx][cury]
            if push_count < 8:
                ns = get_neighbors(a, curx, cury)
                for n in ns:
                    try:
                        if visited[n] == 0:
                            push_count += 1
                            bfsq.append(n)
                    except Exception as ex:
                        print(f'exception {ex} for n {n}')
                        raise ex
        # print(f'no nearest neighbor found after {pop_count} lookups! ...')
        return 0

    def max_vote(annot_img, x, y):
        kernel_size = 5
        votes = defaultdict(int)
        for i in range(x-kernel_size, x+kernel_size):
            for j in range(y-kernel_size, y+kernel_size):
                if i >=0 and i < 512 and j > 0 and j < 512:
                    v = annot_img[i][j]
                    votes[v] += 1
        return max(votes, key=votes.get)
    
    import random
    import time
    
    def do_nn_fill(annot_img):
        print(f'doing nn fill ...')
        start = time.time()
        print(f'zeros {np.sum(annot_img == 0)}')
        for x in range(len(annot_img)):
            for y in range(len(annot_img[0])):
                if annot_img[x][y] == 0:# and random.randint(1,2) == 1:
                    annot_img[x][y] = closest_non_zero(annot_img, x, y)
                    # annot_img[x][y] = max_vote(annot_img, x, y)
        end = time.time()
        print(f'took {end - start} seconds.')
        return annot_img
    
    return do_nn_fill(annot_img)

class LabelPropagate(AbstractHandler):
    def __call__(self,    
        src_img,
        src_depth,
        src_label,
        src_pose,
        cur_pose,
        cur_depth,
    ):
        """
        1. Gets point cloud for the source image 
        2. Transpose the point cloud based on robot location (cur_pose) 
        3. Project the point cloud back into the image frame. The corresponding semantic label for each point from the src_label becomes
        the new semantic label in the current frame.
        Args:
            src_img (np.ndarray): source image to propagte from
            src_depth (np.ndarray): source depth to propagte from
            src_label (np.ndarray): source semantic map to propagte from
            src_pose (np.ndarray): (x,y,theta) of the source image
            cur_pose (np.ndarray): (x,y,theta) of current image
            cur_depth (np.ndarray): current depth
        """
        # Everything assumes image is still vertical

        height, width, _ = src_img.shape
        print(f'height {height} width {width}')
        uv_one_in_cam, intrinsic_mat, rot, trans = compute_uvone(640, 480)
        print(f'uv_one_in_cam.shape {uv_one_in_cam.shape}')

        rgbd = compute_pcd(src_img, src_depth, rot, trans, src_pose, uv_one_in_cam)
        src_pts_in_world = rgbd.ptcloud
        # src_pts_in_world = convert_depth_to_pcd(src_depth, src_pose, uv_one_in_cam, rot, trans)
        
        # visualize using o3d
        # visualize_pcd(src_pts_in_world)
        
        # convert pts_in_world to current base
        src_pts_in_cur_base = transform_pose(src_pts_in_world, (-cur_pose[0], -cur_pose[1], 0))
        src_pts_in_cur_base = transform_pose(src_pts_in_cur_base, (0.0, 0.0, -cur_pose[2]))
            
        # conver point from current base to current camera frame
        src_pts_in_cur_cam = src_pts_in_cur_base - trans.reshape(-1)
        src_pts_in_cur_cam = np.dot(src_pts_in_cur_cam, rot)
        
        # Get Valid Z
        valid_z = src_pts_in_cur_cam[:,2] > 0
        
        # Filter based on current depth.
        # cur_depth = (cur_depth.astype(np.float32) / 1000.0).reshape(-1)
        # cur_pts_in_cur_cam = np.multiply(uv_one_in_cam, cur_depth).T 
        
        # conver pts in current camera frame into 2D pix values
        # src_pts_in_cur_img = np.matmul(intrinsic_mat, src_pts_in_cur_cam.T).T        
        # src_pts_in_cur_img /= src_pts_in_cur_img[:, 2].reshape([-1, 1])

        rgbd = RGBDepth(src_img, cur_depth, src_pts_in_cur_cam)

        uncompute_img = uncompute_pcd(rgbd, rot, trans, cur_pose, uv_one_in_cam)

        # from PIL import Image
        # import matplotlib.pyplot as plt 

        # arr = [cur_img]
        # semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        # semantic_img.putpalette(d3_40_colors_rgb.flatten())
        # semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        # semantic_img = semantic_img.convert("RGBA")
        # arr.append(semantic_img)

        
        return get_annot(height, width, uncompute_img, src_pts_in_cur_cam, None, src_label, valid_z), uncompute_img

