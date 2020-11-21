"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import torch
import torch.utils.data
import torchvision

from box_ops import box_cxcywh_to_xyxy
from datasets.lvis import LvisDetectionBase
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds.get_in_coco_format(img_idx)
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        height, width = targets["orig_size"].tolist()
        img_dict["height"] = height
        img_dict["width"] = width
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        # the boxes are in 0-1 format, in cxcywh format
        # let's convert it into the format expected by COCO api
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes = bboxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, LvisDetectionBase):
        return dataset.lvis
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class PrepareInstance(object):
    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __call__(self, image, target):
        anno = target["annotation"]
        h, w = anno["size"]["height"], anno["size"]["width"]
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno["object"]
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj["bndbox"]
            bbox = [int(bbox[n]) - 1 for n in ["xmin", "ymin", "xmax", "ymax"]]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj["name"]))
            iscrowd.append(int(obj["difficult"]))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno["filename"][:-4]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        # useful metadata
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
