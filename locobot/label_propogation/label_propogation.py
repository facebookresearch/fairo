# used for visualizing how well the label propogation will work
# currenlty will use robot trajectory to realize
# will rely on habitat dat to work on it
# steps we wil perform
# 1. Get point cloud
# 2. Transpose the point cloud based on robot location
# 3. Project the point cloud back the images
import numpy as np
import os
import cv2
import json
from copy import deepcopy as copy
from IPython import embed
import sys

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(BASE_AGENT_ROOT)
from locobot.agent.locobot_mover_utils import transform_pose


root_path = "/checkpoint/dhirajgandhi/active_vision/habitat_data"
image_range = [6875, 7350]  # images in which we can see the object
with open(os.path.join(root_path, "data.json"), "r") as f:
    base_pose_data = json.load(f)

src_img_indx = 6875
src_img = cv2.imread(os.path.join(root_path, "rgb/{:05d}.jpg".format(src_img_indx)))
src_depth = np.load(
    os.path.join(root_path, "depth/{:05d}.npy".format(src_img_indx))
)  # depth is in mm
src_pose = base_pose_data["{}".format(src_img_indx)]
src_label = cv2.imread(os.path.join(root_path, "label/{:05d}.png".format(src_img_indx)))

# TODO: proper entries
intrinsic_mat = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
rot = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
trans = np.array([0, 0, 0.6])

intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
height, width, channels = src_img.shape
img_resolution = (height, width)
img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
img_pixs = img_pixs.reshape(2, -1)
img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

depth = src_depth.astype(np.float32)
depth /= 1000.0
depth = depth.reshape(-1)
pts_in_cam = np.multiply(uv_one_in_cam, depth)
pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
pts_in_base = pts_in_cam[:3, :].T
pts_in_base = np.dot(pts_in_base, rot.T)
pts_in_base = pts_in_base + trans.reshape(-1)
pts_in_world = transform_pose(pts_in_base, src_pose)

# get the selcectd points out of it based on image region
# refer this https://www.codepile.net/pile/bZqJbyNz
indx = zip(*np.where(src_label == 255))
indx = [i[0] * width + i[1] for i in indx]

# not sure if this will work
req_pts_in_world = pts_in_world[indx]

for img_indx in range(image_range[0], image_range[1]):
    print("img_index = {}".format(img_indx))
    # convert the point from world to image frmae
    base_pose = base_pose_data[str(img_indx)]

    # convert point cloud from world pose to base pose
    pts_in_cur_base = copy(req_pts_in_world)
    pts_in_cur_base = transform_pose(pts_in_cur_base, (-base_pose[0], -base_pose[1], 0))
    pts_in_cur_base = transform_pose(pts_in_cur_base, (0.0, 0.0, -base_pose[2]))

    # conver point from base to camera frame
    pts_in_cur_cam = pts_in_cur_base - trans.reshape(-1)
    pts_in_cur_cam = np.dot(pts_in_cur_cam, rot)

    # conver pts from 3D to 2D
    pts_in_cur_img = np.matmul(intrinsic_mat, pts_in_cur_cam.T).T
    pts_in_cur_img /= pts_in_cur_img[:, 2].reshape([-1, 1])

    ##### trying to replace this things
    pts_in_cur_img = np.concatenate(
        (
            np.concatenate(
                (
                    np.ceil(pts_in_cur_img[:, 0]).reshape(-1, 1),
                    np.ceil(pts_in_cur_img[:, 1]).reshape(-1, 1),
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    np.floor(pts_in_cur_img[:, 0]).reshape(-1, 1),
                    np.floor(pts_in_cur_img[:, 1]).reshape(-1, 1),
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    np.ceil(pts_in_cur_img[:, 0]).reshape(-1, 1),
                    np.floor(pts_in_cur_img[:, 1]).reshape(-1, 1),
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    np.floor(pts_in_cur_img[:, 0]).reshape(-1, 1),
                    np.ceil(pts_in_cur_img[:, 1]).reshape(-1, 1),
                ),
                axis=1,
            ),
        )
    )
    # this was the original part
    pts_in_cur_img = pts_in_cur_img[:, :2].astype(int)

    ########
    # TODO: handle the cases where index goes out of the image
    pts_in_cur_img = pts_in_cur_img[
        np.logical_and(pts_in_cur_img[:, 0] < height, pts_in_cur_img[:, 1] < width)
    ]

    # visualize 2D points on image
    # TODO: make sure of the index
    img = cv2.imread(os.path.join(root_path, "rgb/{:05d}.jpg".format(img_indx)))
    """
    img[pts_in_cur_img[:, 1], pts_in_cur_img[:, 0], pts_in_cur_img.shape[0] * [2]] = np.clip(
        img[pts_in_cur_img[:, 1], pts_in_cur_img[:, 0], pts_in_cur_img.shape[0] * [2]] + 30,
        0,
        255,
    )
    """
    img[pts_in_cur_img[:, 1], pts_in_cur_img[:, 0], pts_in_cur_img.shape[0] * [2]] = 255

    # store the image
    cv2.imwrite(os.path.join(root_path, "pred_label/{:05d}.jpg".format(img_indx)), img)
    """
    cv2.imwrite("test_2.jpg", img)
    break
    """

