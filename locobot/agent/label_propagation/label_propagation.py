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
import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import json
from copy import deepcopy as copy
from IPython import embed
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
    print(f" root {root_path}, out {out_dir}, p {propogation_step}")
    ### load the inputs ###
    # load robot trajecotry data which has pose information coreesponding to each img observation taken
    with open(os.path.join(root_path, "data.json"), "r") as f:
        base_pose_data = json.load(f)
    # load img
    try:
        src_img = cv2.imread(os.path.join(root_path, "rgb/{:05d}.jpg".format(src_img_indx)))
        # load depth in mm
        src_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(src_img_indx)))
        # load robot pose for img index
        src_pose = base_pose_data["{}".format(src_img_indx)]
    except:
        print(f"Couldn't load index {src_img_indx} from {root_path}")
        return

    ### data needed to convert depth img to pointcloud ###
    # values extracted from pyrobot habitat agent
    intrinsic_mat = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
    rot = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    trans = np.array([0, 0, 0.6])
    # precompute some values necessary to depth to point cloud
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    height, width, channels = src_img.shape
    img_resolution = (height, width)
    img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
    img_pixs = img_pixs.reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

    ### calculate point cloud in different frmaes ###
    # point cloud in camera frmae
    depth = (src_depth.astype(np.float32) / 1000.0).reshape(-1)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
    # point cloud in robot base frame
    pts_in_base = pts_in_cam[:3, :].T
    pts_in_base = np.dot(pts_in_base, rot.T)
    pts_in_base = pts_in_base + trans.reshape(-1)
    # point cloud in world frame (pyrobot)
    pts_in_world = transform_pose(pts_in_base, src_pose)

    ### figure out unique label values in provided gt label which is greater than 0 ###
    unique_pix_value = np.unique(src_label.reshape(-1), axis=0)
    unique_pix_value = [i for i in unique_pix_value if np.linalg.norm(i) > 0]

    ### for each unique label, figure out points in world frame ###
    # first figure out pixel index
    indx = [zip(*np.where(src_label == i)) for i in unique_pix_value]
    # convert pix index to index in point cloud
    # refer this https://www.codepile.net/pile/bZqJbyNz
    indx = [[i[0] * width + i[1] for i in j] for j in indx]
    # take out points in world space correspoinding to each unique label
    req_pts_in_world_list = [pts_in_world[indx[i]] for i in range(len(indx))]

    # images in which in which we want to label propogation based on the provided gt seg label
    image_range = [max(src_img_indx - propogation_step, 0), src_img_indx + propogation_step]
    # param usful to search nearest point cloud in a region
    kernal_size = 3

    for img_indx in range(image_range[0], image_range[1] + 1):
        print("img_index = {}".format(img_indx))

        ### create point cloud in wolrd frame for img_indx ###
        
        try:
            # get the robot pose value
            base_pose = base_pose_data[str(img_indx)]
            # get the depth
            cur_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(img_indx)))
        except:
            print(f'{img_indx} out of bounds! Total images {len(os.listdir(os.path.join(root_path, "rgb")))}')
            continue
        # convert depth to point cloud in camera frmae
        cur_depth = (cur_depth.astype(np.float32) / 1000.0).reshape(-1)
        cur_pts_in_cam = np.multiply(uv_one_in_cam, cur_depth)
        cur_pts_in_cam = np.concatenate(
            (cur_pts_in_cam, np.ones((1, cur_pts_in_cam.shape[1]))), axis=0
        )
        # convert point cloud in camera fromae to base frame
        cur_pts_in_base = cur_pts_in_cam[:3, :].T
        cur_pts_in_base = np.dot(cur_pts_in_base, rot.T)
        cur_pts_in_base = cur_pts_in_base + trans.reshape(-1)
        # convert point cloud from base frame to world frmae
        cur_pts_in_world = transform_pose(cur_pts_in_base, base_pose)

        ### generate label for new img indx ###
        # crete annotation files with all zeros
        annot_img = np.zeros((height, width))
        # do label prpogation for each unique label in provided gt seg label
        for i, (req_pts_in_world, pix_color) in enumerate(
            zip(req_pts_in_world_list, unique_pix_value)
        ):
            # convert point cloud for label from world pose to current (img_indx) base pose
            pts_in_cur_base = copy(req_pts_in_world)
            pts_in_cur_base = transform_pose(pts_in_cur_base, (-base_pose[0], -base_pose[1], 0))
            pts_in_cur_base = transform_pose(pts_in_cur_base, (0.0, 0.0, -base_pose[2]))

            # conver point from current base to current camera frame
            pts_in_cur_cam = pts_in_cur_base - trans.reshape(-1)
            pts_in_cur_cam = np.dot(pts_in_cur_cam, rot)

            # conver pts in current camera frame into 2D pix values
            pts_in_cur_img = np.matmul(intrinsic_mat, pts_in_cur_cam.T).T
            pts_in_cur_img /= pts_in_cur_img[:, 2].reshape([-1, 1])

            # filter out index which fall beyond the shape of img size
            filtered_img_indx = np.logical_and(
                np.logical_and(0 <= pts_in_cur_img[:, 0], pts_in_cur_img[:, 0] < height),
                np.logical_and(0 <= pts_in_cur_img[:, 1], pts_in_cur_img[:, 1] < width),
            )

            # only consider depth matching for these points
            # filter out point based on projected depth value wold frmae, this helps us get rid of pixels for which view to the object is blocked by any other object, as in that was projected 3D point in wolrd frmae for the current pix wont match with 3D point in the gt provide label
            dist_thr = 5e-2  # this is in meter
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
                    loc = loc.reshape(-1).astype(np.int32)
                    loc = np.clip(loc, 0, width * height - 1).astype(np.int32)

                    if (
                        min(np.linalg.norm(cur_pts_in_world[loc] - gt_pix_depth_in_world, axis=1))
                        > dist_thr
                    ):
                        filtered_img_indx[pixel_index] = False

            # take out filtered pix values
            pts_in_cur_img = pts_in_cur_img[filtered_img_indx]

            # step to take care of quantization erros
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
            pts_in_cur_img = pts_in_cur_img[:, :2].astype(int)

            # filter out index which fall beyonf the shape of img size, had to perform this step again to take care if any out of the image size point is introduced by the above quantization step
            pts_in_cur_img = pts_in_cur_img[
                np.logical_and(
                    np.logical_and(0 <= pts_in_cur_img[:, 0], pts_in_cur_img[:, 0] < height),
                    np.logical_and(0 <= pts_in_cur_img[:, 1], pts_in_cur_img[:, 1] < width),
                )
            ]

            # number of pointf for the label found in cur img
            # print("pts in cam = {}".format(len(pts_in_cur_cam)))

            # assign label to correspoinding pix values
            annot_img[pts_in_cur_img[:, 1], pts_in_cur_img[:, 0]] = pix_color

        # store the annotation file
        np.save(os.path.join(out_dir, "{:05d}.npy".format(img_indx)), annot_img.astype(np.uint32))


def run_label_prop(out_dir, gtframes, propagation_step, root_path):
    start = time.time()
    # load the file for train images to be used for label propogation
    # root_path = data_path
    # out_dir = args.out_dir #os.path.join(root_path, args.out_dir)
    ray.shutdown()
    # use all avialeble cpus -1
    ray.init(num_cpus=os.cpu_count() - 1)
    result_ids = []
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(root_path, "data.json"), "r") as f:
        base_pose_data = json.load(f)

    num_imgs = len(glob.glob(os.path.join(root_path, "rgb/*.jpg")))
    result = []
    train_img_id = {"img_id": []}
    src_img_indx = 0
    delta = int(num_imgs / gtframes)
    propagation_step = min(propagation_step, int(delta/2))
    train_img_id['propagation_step'] = propagation_step
    
    for x in range(gtframes):
        src_img_indx += delta
        if os.path.isfile(os.path.join(root_path, "seg/{:05d}.npy".format(src_img_indx))):
            result.append(
                propogate_label.remote(
                    root_path=root_path,
                    src_img_indx=src_img_indx,
                    src_label=np.load(
                        os.path.join(root_path, "seg/{:05d}.npy".format(src_img_indx))
                    ),
                    propogation_step=propagation_step,
                    base_pose_data=base_pose_data,
                    out_dir=out_dir,
                )
            )
            train_img_id["img_id"].append(src_img_indx)

    with open(os.path.join(out_dir, "train_img_id.json"), "w") as fp:
        json.dump(train_img_id, fp)
    ray.get(result)
    print("duration =", time.time() - start)

    run_metrics(out_dir, root_path)


def run_metrics(out_dir, root_path):
    # Measure accuracy
    pass
