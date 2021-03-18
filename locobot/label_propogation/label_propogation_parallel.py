# used for visualizing how well the label propogation will work
# currenlty will use robot trajectory to realize
# will rely on habitat dat to work on it

# steps we will perform for label propogation
# 1. Get point cloud
# 2. Transpose the point cloud based on robot location
# 3. Project the point cloud back the images
from tokenize import String
import numpy as np
import os
import cv2
import json
from copy import deepcopy as copy
from IPython import embed
import sys
import time
import ray
from scipy.spatial.transform import Rotation
from pycocotools.coco import COCO
import glob
import argparse


# this function is implemented at 'from locobot.agent.locobot_mover_utils import transform_pose'
# however ray was having toruble finding the function
def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
    current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1, 3))
    XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
    XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
    return XYZ


@ray.remote
def propogate_label(
    root_path: str,
    src_img_indx: int,
    src_label: np.ndarray,
    propogation_step: int,
    base_pose_data: np.ndarray,
    out_dir: str,
):
    """Take the label for src_img_indx and propogate it to [src_img_indx - propogation_step, src_img_indx + propogation_step]
    Args:
        root_path (str): root path where images are stored
        src_img_indx (int): source image index
        src_label (np.ndarray): array with labeled images are stored (hwc format)
        propogation_step (int): number of steps to progate the label
        base_pose_data(np.ndarray): (x,y,theta)
        out_dir (str): path to store labeled propogation image
    """

    # images in which we can see the object
    image_range = [max(src_img_indx - propogation_step, 0), src_img_indx + propogation_step]

    with open(os.path.join(root_path, "data.json"), "r") as f:
        base_pose_data = json.load(f)

    src_img = cv2.imread(os.path.join(root_path, "rgb/{:05d}.jpg".format(src_img_indx)))
    src_depth = np.load(
        os.path.join(root_path, "depth/{:05d}.npy".format(src_img_indx))
    )  # depth is in mm
    src_pose = base_pose_data["{}".format(src_img_indx)]

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

    depth = (src_depth.astype(np.float32) / 1000.0).reshape(-1)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
    pts_in_base = pts_in_cam[:3, :].T
    pts_in_base = np.dot(pts_in_base, rot.T)
    pts_in_base = pts_in_base + trans.reshape(-1)
    pts_in_world = transform_pose(pts_in_base, src_pose)

    # get the selcectd points out of it based on image region
    # refer this https://www.codepile.net/pile/bZqJbyNz
    # TODO: Make it work with labels containing multiple label
    unique_pix_value = np.unique(src_label.reshape(-1), axis=0)
    unique_pix_value = [i for i in unique_pix_value if np.linalg.norm(i) > 0]

    indx = [zip(*np.where(src_label == i)) for i in unique_pix_value]
    indx = [[i[0] * width + i[1] for i in j] for j in indx]

    # not sure if this will work
    req_pts_in_world_list = [pts_in_world[indx[i]] for i in range(len(indx))]

    kernal_size = 3
    for img_indx in range(image_range[0], image_range[1]):
        print("img_index = {}".format(img_indx))

        # convert the point from world to image frmae
        base_pose = base_pose_data[str(img_indx)]
        try:
            cur_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(img_indx)))
        except:
            return

        cur_depth = (cur_depth.astype(np.float32) / 1000.0).reshape(-1)
        cur_pts_in_cam = np.multiply(uv_one_in_cam, cur_depth)
        cur_pts_in_cam = np.concatenate(
            (cur_pts_in_cam, np.ones((1, cur_pts_in_cam.shape[1]))), axis=0
        )
        cur_pts_in_base = cur_pts_in_cam[:3, :].T
        cur_pts_in_base = np.dot(cur_pts_in_base, rot.T)
        cur_pts_in_base = cur_pts_in_base + trans.reshape(-1)
        cur_pts_in_world = transform_pose(cur_pts_in_base, base_pose)

        annot_img = np.zeros((height, width))

        for i, (req_pts_in_world, pix_color) in enumerate(
            zip(req_pts_in_world_list, unique_pix_value)
        ):
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

            # only consider depth matching for these points
            filtered_img_indx = np.logical_and(
                np.logical_and(0 <= pts_in_cur_img[:, 0], pts_in_cur_img[:, 0] < height),
                np.logical_and(0 <= pts_in_cur_img[:, 1], pts_in_cur_img[:, 1] < width),
            )

            start_time = time.time()
            # TODO: make this part fast, its very slow currently
            dist_thr = 5e-2  # this is in meter
            ## optimize part
            start_time = time.time()
            ## need to optimize this part
            for pixel_index in range(len(filtered_img_indx)):
                if filtered_img_indx[pixel_index]:
                    # search in the region
                    gt_pix_depth_in_world = req_pts_in_world[pixel_index]
                    p, q = np.meshgrid(
                        range(
                            int(pts_in_cur_img[pixel_index][1] - kernal_size),
                            int(pts_in_cur_img[pixel_index][1] + kernal_size),
                        ),
                        range(
                            int(pts_in_cur_img[pixel_index][0] - kernal_size),
                            int(pts_in_cur_img[pixel_index][0] + kernal_size),
                        ),
                    )
                    loc = p * width + q
                    loc = loc.reshape(-1).astype(np.int)
                    loc = np.clip(loc, 0, width * height - 1).astype(np.int)

                    if (
                        min(np.linalg.norm(cur_pts_in_world[loc] - gt_pix_depth_in_world, axis=1))
                        > dist_thr
                    ):
                        filtered_img_indx[pixel_index] = False

            # take out the points
            pts_in_cur_img = pts_in_cur_img[filtered_img_indx]
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
                np.logical_and(
                    np.logical_and(0 <= pts_in_cur_img[:, 0], pts_in_cur_img[:, 0] < height),
                    np.logical_and(0 <= pts_in_cur_img[:, 1], pts_in_cur_img[:, 1] < width),
                )
            ]
            print("pts in cam = {}".format(len(pts_in_cur_cam)))
            annot_img[pts_in_cur_img[:, 1], pts_in_cur_img[:, 0]] = pix_color

        # store the value
        np.save(os.path.join(out_dir, "{:05d}.npy".format(img_indx)), annot_img.astype(np.uint32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for testing simple SLAM algorithm")
    parser.add_argument(
        "--scene_path",
        help="path where scene data is being stored",
        type=str,
        default="/checkpoint/dhirajgandhi/active_vision/replica_random_exploration_data",
    )
    parser.add_argument("--freq", help="freq to use ground truth seg label", type=int, default=30)
    parser.add_argument(
        "--propogation_step",
        help="number of steps till porpgate label (both +ve and -ve side)",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--out_dir",
        help="path where to store label propogation data inside scene folder",
        type=str,
        default="pred_label_using_traj",
    )
    args = parser.parse_args()
    # code assumes the following structure of data
    """
    ├── scene_path
    │   ├── apartment_0
    │   │   ├── rgb
    |   │   │   ├── 00000.jpg
    |   │   │   ├── 00001.jpg
                .
                .
    │   │   ├── seg
    |   │   │   ├── 00000.npy
    |   │   │   ├── 00001.npy
                .
                .
    │   │   ├── out_dir
    |   │   │   ├── 00000.npy
    |   │   │   ├── 00001.npy
                .
                .
    │   │   ├── data.json (robot state information with corresponding image id)
        .
        .
    │   ├── apartment_1 
        .
        .     
    """

    start = time.time()
    # load the file for train images to be used for label propogation
    scene_stored_path = args.scene_path
    for scene in os.listdir(scene_stored_path):
        root_path = os.path.join(scene_stored_path, scene)
        out_dir = os.path.join(root_path, args.out_dir)
        ray.shutdown()
        ray.init(num_cpus=79)
        result_ids = []
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(root_path, "data.json"), "r") as f:
            base_pose_data = json.load(f)

        num_imgs = len(glob.glob(os.path.join(root_path, "rgb/*.jpg")))
        propogation_step = args.propogation_step
        result = [
            propogate_label.remote(
                root_path=root_path,
                src_img_indx=src_img_indx,
                src_label=np.load(os.path.join(root_path, "seg/{:05d}.npy".format(src_img_indx))),
                propogation_step=propogation_step,
                base_pose_data=base_pose_data,
                out_dir=out_dir,
            )
            for src_img_indx in range(0, num_imgs - propogation_step, args.freq)
        ]
        ray.get(result)
    print("duration =", time.time() - start)
