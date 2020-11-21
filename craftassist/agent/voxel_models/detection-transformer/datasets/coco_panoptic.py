"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

import transforms as T
from box_ops import masks_to_boxes
from panopticapi.utils import rgb2id


class CocoPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None):
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        for img, ann in zip(self.coco["images"], self.coco["annotations"]):
            assert img["file_name"][:-4] == ann["file_name"][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transforms

    def __getitem__(self, idx):
        ann_info = self.coco["annotations"][idx]
        img_path = os.path.join(self.img_folder, ann_info["file_name"].replace(".png", ".jpg"))
        ann_path = os.path.join(self.ann_folder, ann_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
        masks = rgb2id(masks)

        ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
        masks = masks == ids[:, None, None]

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.tensor(
            [ann["category_id"] for ann in ann_info["segments_info"]], dtype=torch.int64
        )

        target = {}
        target["image_id"] = torch.tensor([ann_info["image_id"]])
        target["masks"] = masks
        target["labels"] = labels
        w, h = img.size

        target["boxes"] = masks_to_boxes(masks)

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        for name in ["iscrowd", "area"]:
            target[name] = torch.tensor([ann[name] for ann in ann_info["segments_info"]])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # print('img')
        # print(img)
        # print('target')
        # print(target)
        return img, target

    def __len__(self):
        return len(self.coco["images"])

    def get_height_and_width(self, idx):
        img_info = self.coco["images"][idx]
        height = img_info["height"]
        width = img_info["width"]
        return height, width


def make_coco_panoptic_transforms(image_set):

    normalize = T.Compose(
        [
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    transform_train = T.Compose(
        [
            # T.RandomHorizontalFlip(),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize
        ]
    )

    transform_val = T.Compose([T.RandomResize([800], max_size=1333), normalize])

    transforms = {
        "train": transform_train,
        "trainval": transform_train,
        "val": transform_val,
        "test": transform_val,
    }

    return transforms[image_set]


def build(image_set, args):
    img_folder_root = "/datasets01/COCO/022719"
    ann_folder_root = "/datasets01/COCO/060419"
    mode = "panoptic"
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": (
            "train2017",
            os.path.join("annotations", anno_file_template.format(mode, "train")),
        ),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # this is a hack, change in the future
        "trainval": (
            "train2017",
            os.path.join("annotations", anno_file_template.format(mode, "train")),
        ),
        "test": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = os.path.join(img_folder_root, img_folder)
    ann_folder = os.path.join(ann_folder_root, "{}_{}".format(mode, img_folder))
    ann_file = os.path.join(ann_folder_root, ann_file)

    dataset = CocoPanoptic(
        img_folder_path, ann_folder, ann_file, transforms=make_coco_panoptic_transforms(image_set)
    )

    return dataset
