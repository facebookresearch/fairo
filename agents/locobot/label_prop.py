"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import base64
import cv2
from imantics import Mask, Polygons
import numpy as np
from PIL import Image
from pathlib import Path
import json
from pycococreatortools import pycococreatortools
from sklearn.model_selection import train_test_split
import glob
import random
import torch

from detectron2.structures.boxes import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from droidlet.perception.robot.detectron.detector.dataset_mapper import LocobotDatasetMapper
from droidlet.perception.robot import LabelPropagate

def label_propagation(postData): 
    """
    postData: 
        prevRgbImg: source rgb image
        prevDepth: source depth object
            depthImg: depth map image
            depthMax: maximum value in original depth map
            depthMin: minimum value in original depth map
        depth: current depth object
            same properties as prevDepth
        objects: array of objects
        prevBasePose: source base pose data
        basePose: current base pose data

    Returns a dictionary mapping objects ids to the objects with new masks
    """

    # Decode rgb map
    rgb_bytes = base64.b64decode(postData["prevRgbImg"])
    rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
    rgb = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    src_img = np.array(rgb)
    height, width, _ = src_img.shape

    # Convert depth map to meters
    depth_imgs = []
    for i, depth in enumerate([postData["prevDepth"], postData["depth"]]): 
        depth_encoded = depth["depthImg"]
        depth_bytes = base64.b64decode(depth_encoded)
        depth_np = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_decoded = cv2.imdecode(depth_np, cv2.IMREAD_COLOR)
        depth_unscaled = (255 - np.copy(depth_decoded[:,:,0]))
        depth_scaled = depth_unscaled / 255 * (float(depth["depthMax"]) - float(depth["depthMin"]))
        depth_imgs.append(depth_scaled + float(depth["depthMin"]))
    src_depth = np.array(depth_imgs[0])
    cur_depth = np.array(depth_imgs[1])

    # Labels map
    src_label = mask_to_map(postData["objects"], height, width)

    # Attach base pose data
    pose = postData["prevBasePose"]
    src_pose = np.array([pose["x"], pose["y"], pose["yaw"]])
    pose = postData["basePose"]
    cur_pose = np.array([pose["x"], pose["y"], pose["yaw"]])

    LP = LabelPropagate()
    res_labels = LP(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)

    # Convert mask maps to mask points
    objects = labels_to_objects(res_labels, postData["objects"])

    # Returns an array of objects with updated masks
    return objects

# LP helper: Convert mask points to mask maps then combine them
def mask_to_map(objects, height, width): 
    res = np.zeros((height, width)).astype(int)
    for n, o in enumerate(objects): 
        poly = Polygons(o["mask"])
        bitmap = poly.mask(height, width)
        for i in range(height): 
            for j in range(width): 
                if bitmap[i][j]: 
                    res[i][j] = n + 1
    return res

# LP helper: Convert mask maps to mask points in object structure
def labels_to_objects(labels, objects): 
    res = {}
    for i_float in np.unique(labels): 
        i = int(i_float)
        if i == 0: 
            continue
        res[i-1] = objects[i-1] # Do this in the for loop cause some objects aren't returned
        mask_points_nd = Mask(np.where(labels == i, 1, 0)).polygons().points
        mask_points = list(map(lambda x: x.tolist(), mask_points_nd))
        res[i-1]["mask"] = mask_points
        res[i-1]["type"] = "annotate"
    return res

def save_rgb_seg(postData): 
    """
    postData: 
        rgb: rgb image to be saved
        categories: array starting with null of categories saved in dashboard
        objects: array of objects with masks and labels
        frameCount: counter to help name output files

    Saves rgb image into annotation_data/rgb and creates a segmentation map to
    be saved in annotation_data/seg. 
    """

    # Decode rgb map
    rgb_bytes = base64.b64decode(postData["rgb"])
    rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
    rgb_bgr = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    src_img = np.array(rgb)
    height, width, _ = src_img.shape

    # Properties
    frameId = str(postData["frameCount"])
    if "props.json" in os.listdir("annotation_data"): 
        with open("annotation_data/props.json") as f:
            props = json.load(f)
    else: 
        props = {}
    props[frameId] = {}

    # Convert mask points to mask maps then combine them
    categories = postData["categories"]
    display_map = np.zeros((height, width)).astype(int) # 2 separate chair masks will be same color here
    for o in postData["objects"]: 
        poly = Polygons(o["mask"])
        bitmap = poly.mask(height, width)
        index = categories.index(o["label"])
        for i in range(height): 
            for j in range(width): 
                if bitmap[i][j]: 
                    display_map[i][j] = index
        # Add properties
        o_props = o["properties"].split("\n ")
        if index in props[frameId]: 
            prop_set = set(props[frameId][index])
            prop_set.update(o_props)
            props[frameId][index] = list(prop_set)
        else: 
            props[frameId][index] = o_props

    # Save annotation data to disk for retraining
    Path("annotation_data/seg").mkdir(parents=True, exist_ok=True)
    Path("annotation_data/rgb").mkdir(parents=True, exist_ok=True)
    np.save("annotation_data/seg/{:05d}.npy".format(postData["frameCount"]), display_map)
    im = Image.fromarray(src_img)
    im.save("annotation_data/rgb/{:05d}.jpg".format(postData["frameCount"]))
    with open("annotation_data/props.json", "w") as file: 
        json.dump(props, file)

def save_annotations(categories, properties): 
    """
    categories: array starting with null of categories saved in dashboard

    Saves the annotations in annotation/seg/ and annotation/rgb/ into COCO 
    format for use in retraining. The resulting file is located at the filepath
    for coco_file_name. 
    """

    seg_dir = "annotation_data/seg/"
    img_dir = "annotation_data/rgb/"
    coco_file_name = "annotation_data/coco/coco_results.json"

    fs = [x.split(".")[0] + ".jpg" for x in os.listdir(seg_dir)]

    INFO = {}
    LICENSES = [{}]
    CATEGORIES = []
    id_to_label = {}
    for i, label in enumerate(categories):
        if not label: 
            continue
        CATEGORIES.append({"id": i, "name": label, "supercategory": "shape"})
        id_to_label[i] = label

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
        "properties": properties,
    }

    count = 0
    for x in fs:
        image_id = int(x.split(".")[0])
        # load the annotation file
        try:
            seg_path = os.path.join(seg_dir, "{:05d}.npy".format(image_id))
            annot = np.load(seg_path, allow_pickle=True).astype(np.uint8)
        except Exception as e:
            print(e)
            continue

        img_filename = "{:05d}.jpg".format(image_id)
        img = Image.open(os.path.join(img_dir, img_filename))
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(img_filename), img.size
        )

        coco_output["images"].append(image_info)

        # for each annotation add to coco format
        for i in np.sort(np.unique(annot.reshape(-1), axis=0)):
            try:
                category_info = {"id": int(i), "is_crowd": False}
                if category_info["id"] < 1:
                    continue
            except:
                print("label value doesnt exist for", i)
                continue
            binary_mask = (annot == i).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                count, image_id, category_info, binary_mask, img.size, tolerance=2
            )
            if annotation_info is not None:
                annotation_info["properties"] = []
                coco_output["annotations"].append(annotation_info)
                count += 1

    Path("annotation_data/coco").mkdir(parents=True, exist_ok=True)
    with open(coco_file_name, "w") as output_json:
        json.dump(coco_output, output_json)
        print("Saved annotations to", coco_file_name)

def save_categories_properties(categories, properties): 
    """
    categories: array starting with null of categories saved in dashboard
    properties: array of properties used to describe objects

    Adds the new categories and properties to the json files that already 
    exist. The updated categories and properties are stored in things_path
    and props_path, respectively. 
    """

    # Create new versioned models folder
    models_dir = "annotation_data/model"
    model_files = os.listdir(models_dir)
    # Get highest numbered x for model/vx directories
    model_dirs = list(filter(lambda n: os.path.isdir(os.path.join(models_dir, n)), model_files))
    model_nums = list(map(lambda x: int(x.split("v")[1]), model_dirs))
    cur_model_num = max(model_nums) + 1
    # Create new folder for model/v(x+1)
    model_dir = os.path.join(models_dir, "v" + str(cur_model_num))
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Get categories and properties
    things_path = os.path.join(model_dir, "things.json")
    cats = set(categories[1:])
    cats_json = {
        "items": list(cats), 
    }
    props_path = os.path.join(model_dir, "props.json")
    props = set(properties)
    props_json = {
        "items": list(props), 
    }

    # Write to file
    with open(things_path, "w") as file: 
        json.dump(cats_json, file)
        print("saved categories to", things_path)
    with open(props_path, "w") as file: 
        json.dump(props_json, file)
        print("saved properties to", props_path)

# Used for retraining
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = LocobotDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

def retrain_detector(settings): 
    """
    settings: properties to be used in the retraining process

    Splits the COCO-formatted data located in annotation_path, then trains and 
    evaluates a Detectron2 model from scratch. The resulting model is saved in 
    the model_path/ folder. 

    Returns an object mapping different AP (average precision) metrics to the 
    model's scores. 
    """

    if len(settings) == 0: 
        settings["trainSplit"] = 0.7
        settings["learningRate"] = 0.005
        settings["maxIters"] = 100

    base_path = "annotation_data/"
    coco_path = os.path.join(base_path, "coco")
    output_path = os.path.join(base_path, "output")
    annotation_path = os.path.join(coco_path, "coco_results.json")
    train_path = os.path.join(coco_path, "train.json")
    test_path = os.path.join(coco_path, "test.json")

    # 1) Split coco json file into train and test using cocosplit code
    # Adapted from https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py
    with open(annotation_path, "rt", encoding="UTF-8") as annotations_file: 
        
        # Extract info from json
        coco = json.load(annotations_file)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]
        properties = coco["properties"]

        # Remove images without annotations
        images_with_annotations = set(map(lambda a: int(a["image_id"]), annotations))
        images = list(filter(lambda i: i["id"] in images_with_annotations, images))

        # Split images and annotations
        x_images, y_images = train_test_split(images, train_size=settings["trainSplit"])
        x_ids = list(map(lambda i: int(i["id"]), x_images))
        x_annots = list(filter(lambda a: int(a["image_id"]) in x_ids, annotations))
        y_ids = list(map(lambda i: int(i["id"]), y_images))
        y_annots = list(filter(lambda a: int(a["image_id"]) in y_ids, annotations))

        # Save to file
        def save_coco(file, info, licenses, images, annotations, categories): 
            with open(file, 'wt', encoding="UTF-8") as coco: 
                json.dump({ 
                    "info": info, 
                    "licenses": licenses, 
                    "images": images, 
                    "annotations": annotations, 
                    "categories": categories,
                }, coco, indent=2, sort_keys=True)
        save_coco(train_path, info, licenses, x_images, x_annots, categories)
        save_coco(test_path, info, licenses, y_images, y_annots, categories)

    # 2) Use train/test files to retrain detector
    dataset_name = "annotation_coco"
    image_dir = base_path + "rgb/"
    train_data = dataset_name + "_train"
    test_data = dataset_name + "_test"

    # Builds dict to be used in DatasetCatalog
    def get_dataset_dicts(json_file, props_dict):
        with open(json_file) as f: 
            coco_info = json.load(f)
        annots = coco_info["annotations"]
        images = coco_info["images"]

        dataset_dicts = []
        for _, img in enumerate(images):
            record = {}
            dirname = os.path.dirname(os.path.realpath(__file__))
            relative_rgb_path = "../../annotation_data/rgb"
            rgb_path = os.path.join(dirname, relative_rgb_path)
            record["file_name"] = os.path.join(rgb_path, img["file_name"])
            record["image_id"] = img["id"]
            record["height"] = img["height"]
            record["width"] = img["width"]
            objs = []
            keep_rec = False
            for _, anno in enumerate(annots):
                if anno["image_id"] == img["id"]:
                    keep_rec = True
                    obj_props = props_dict[str(img["id"])][str(anno["category_id"])]
                    obj = {
                        "bbox": anno["bbox"],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": anno["segmentation"],
                        "category_id": anno["category_id"],
                        "iscrowd": 0,
                        "properties": obj_props,
                    }
                    objs.append(obj)

            record["annotations"] = objs
            if keep_rec:
                dataset_dicts.append(record)
        return dataset_dicts

    # Create DatasetCatalog and MetadataCatalog
    DatasetCatalog.clear()
    with open("annotation_data/props.json") as f: 
        props_dict = json.load(f)
    DatasetCatalog.register(train_data, lambda: get_dataset_dicts(train_path, props_dict))
    DatasetCatalog.register(test_data, lambda: get_dataset_dicts(test_path, props_dict))
    MetadataCatalog.get(train_data).set(
        property_classes=properties, thing_classes=categories, json_file=train_path, image_root=image_dir, evaluator_type="coco"
    )
    MetadataCatalog.get(test_data).set(
        property_classes=properties, thing_classes=categories, json_file=test_path, image_root=image_dir, evaluator_type="coco"
    )

    # Path for model lyaml
    dirname = os.path.dirname(os.path.realpath(__file__))
    coco_yaml = "../../droidlet/perception/robot/detectron/detector/configs/mask_rcnn_R_101_FPN_1x.yaml"
    config_path = os.path.join(dirname, coco_yaml)

     # Set config
    cfg = get_cfg()
    cfg.MODEL.DENSEPOSE_ON = True
    cfg.MODEL.ROI_PROPERTY_HEAD = CN()
    cfg.MODEL.ROI_PROPERTY_HEAD.NAME = ""
    cfg.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES = len(properties)
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = (train_data,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = output_path
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = settings["learningRate"] # Make sure LR is good
    cfg.SOLVER.MAX_ITER = settings["maxIters"] # 300 is good for small datasets
    
    # Train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Move model to most recent model folder
    model_dir = os.path.join(base_path, "model")
    model_names = os.listdir(model_dir)
    # Get highest x for model/vx
    model_dirs = list(filter(lambda n: os.path.isdir(os.path.join(model_dir, n)), model_names))
    model_nums = list(map(lambda x: int(x.split("v")[1]), model_dirs))
    last_model_num = max(model_nums) 
    # Add model to new folder
    model_path = os.path.join(model_dir, "v" + str(last_model_num))
    new_model_path = os.path.join(model_path, "model_999.pth")
    old_model_path = os.path.join(output_path, "model_final.pth")
    os.replace(old_model_path, new_model_path)

    # Evaluate
    evaluator = COCOEvaluator(test_data, ("bbox", "segm"), False, output_dir="../../annotation_data/output/")
    val_loader = build_detection_test_loader(cfg, test_data)
    inference = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    # inference keys: bbox, semg
    # bbox and segm keys: AP, AP50, AP75, APs, APm, AP1, AP-category1, ...
    inference_json = json.loads(json.dumps(inference).replace("NaN", "null"))
    return inference_json

def get_offline_frame(data): 
    """
    data: 
        filepath: where to access rgb data on disk
        frameId: which file to access on disk 

    Reads in the specified rgb image and depth map, then encodes and sends the 
    processed rgb image and depth map. Mimics RGBDepth.to_struct(). 
    """

    rgb_path = os.path.join(data["filepath"], "rgb")
    depth_path = os.path.join(data["filepath"], "depth")

    num_zeros = 5 - len(str(data["frameId"]))
    file_num = "".join(["0" for _ in range(num_zeros)]) + str(data["frameId"])
    rgb_filename = os.path.join(rgb_path, file_num + ".jpg")
    depth_filename = os.path.join(depth_path, file_num + ".npy")

    rgb = cv2.imread(rgb_filename)
    depth = np.load(depth_filename)

    depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_img = 255 - depth_img

    # webp seems to be better than png and jpg as a codec, in both compression and quality
    quality = 10
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    fmt = ".webp"

    _, rgb_data = cv2.imencode(fmt, rgb, encode_param)
    _, depth_img_data = cv2.imencode(fmt, depth_img, encode_param)

    rgb = base64.b64encode(rgb_data).decode("utf-8")
    depth = {
        "depthImg": base64.b64encode(depth_img_data).decode("utf-8"),
        "depthMax": str(np.max(depth)),
        "depthMin": str(np.min(depth)),
    }
    return rgb, depth
