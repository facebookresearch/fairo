"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from droidlet.perception.robot import Detection, Human, HumanKeypointsOrdering
from droidlet.shared_data_structs import RGBDepth


def get_fake_rgbd(rgb=None, h=480, w=640):
    rgb = np.float32(np.random.rand(h, w, 3) * 255) if rgb is None else rgb
    depth = np.random.randint(0, 2, (h, w))
    pts = np.random.randint(0, 5, (h * w, 3))
    return RGBDepth(rgb, depth, pts)


def get_fake_mask(h=480, w=640):
    return np.random.randint(0, 1, (h, w))


def get_fake_bbox(h=480, w=640):
    y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
    y2, x2 = np.random.randint(y1, h), np.random.randint(x1, w)
    return [x1, y1, x2, y2]


def get_fake_detection(class_label, properties, xyz):
    rgb_d = get_fake_rgbd()
    mask = get_fake_mask()
    bbox = get_fake_bbox()
    d = Detection(
        rgb_d,
        class_label=class_label,
        properties=properties,
        mask=mask,
        bbox=bbox,
        xyz=xyz,
        center=(0, 0),
    )
    return d


def get_rand_pixel(rgb_d):
    h = rgb_d.rgb.shape[0]
    w = rgb_d.rgb.shape[1]
    # return in (w,h) format, since that is the currently followed convention for points
    # See https://github.com/facebookresearch/fairo/issues/358
    return [np.random.randint(0, w), np.random.randint(0, h)]


def get_fake_human_keypoints(rgb_d):
    # "keypoints" is a length 3k array where k is the total number of keypoints defined
    # for the category. Each keypoint has a 0-indexed location x,y and a visibility flag
    # v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible,
    # and v=2: labeled and visible. source https://cocodataset.org/#format-data
    return HumanKeypointsOrdering._make(
        [
            get_rand_pixel(rgb_d) + [np.random.randint(0, 3)]
            for x in range(len(HumanKeypointsOrdering._fields))
        ]
    )


def get_fake_humanpose():
    rgb_d = get_fake_rgbd()
    keypoints = get_fake_human_keypoints(rgb_d)
    return Human(rgb_d, keypoints)
