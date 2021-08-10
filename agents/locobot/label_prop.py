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

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

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

    # Save annotation data to disk for retraining
    Path("annotation_data/seg").mkdir(parents=True, exist_ok=True)
    Path("annotation_data/rgb").mkdir(parents=True, exist_ok=True)
    np.save("annotation_data/seg/{:05d}.npy".format(postData["frameCount"]), display_map)
    im = Image.fromarray(src_img)
    im.save("annotation_data/rgb/{:05d}.jpg".format(postData["frameCount"]))

def save_annotations(categories): 
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
    removed_categories = []
    for i, label in enumerate(categories):
        if not label: 
            continue
        CATEGORIES.append({"id": i, "name": label, "supercategory": "shape"})
        id_to_label[i] = label
        if label in ("floor", "wall", "ceiling", "wall-plug"):
            removed_categories.append(i)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
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
                if category_info["id"] < 1 or category_info["id"] in removed_categories:
                    # Exclude wall, ceiling, floor, wall-plug
                    continue
            except:
                print("label value doesnt exist for", i)
                continue
            binary_mask = (annot == i).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                count, image_id, category_info, binary_mask, img.size, tolerance=2
            )
            if annotation_info is not None:
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

    # Load existing categories & properties
    file_dir = "annotation_data/model"
    things_path = os.path.join(file_dir, "things.json")
    if os.path.exists(things_path): 
        with open(things_path, "rt") as file: 
            things_dict = json.load(file)
            cats = set(things_dict["items"])
    else: 
        cats = set()
    props_path = os.path.join(file_dir, "props.json")
    if os.path.exists(props_path): 
        with open(props_path, "rt") as file: 
            props_dict = json.load(file)
            props = set(props_dict["items"])
    else: 
        props = set()

    # Add new categories & properties
    cats.update(categories[1:]) # Don't add null
    cats_json = {
        "items": list(cats), 
    }
    props.update(properties)
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
    coco_path = base_path + "coco/"
    output_path = base_path + "output/"
    model_path = base_path + "model/"
    annotation_path = coco_path + "coco_results.json"
    train_path = coco_path + "train.json"
    test_path = coco_path + "test.json"
    train_split = settings["trainSplit"]

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

        # Remove images without annotations
        images_with_annotations = set(map(lambda a: int(a["image_id"]), annotations))
        images = list(filter(lambda i: i["id"] in images_with_annotations, images))

        # Split images and annotations
        x_images, y_images = train_test_split(images, train_size=train_split)
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
                    "categories": categories
                }, coco, indent=2, sort_keys=True)
        save_coco(train_path, info, licenses, x_images, x_annots, categories)
        save_coco(test_path, info, licenses, y_images, y_annots, categories)

    # 2) Use train/test files to retrain detector
    # Adapted from train_detector.ipynb
    dataset_name = "annotation_coco"
    image_dir = base_path + "rgb/"
    train_data = dataset_name + "_train"
    test_data = dataset_name + "_test"

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    register_coco_instances(train_data, {}, train_path, image_dir)
    register_coco_instances(test_data, {}, test_path, image_dir)

    MetadataCatalog.get(train_data)
    coco_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(coco_yaml))
    cfg.DATASETS.TRAIN = (train_data,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_yaml)  # Let training initialize from model zoo
    cfg.OUTPUT_DIR = output_path
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = settings["learningRate"] # Make sure LR is good
    cfg.SOLVER.MAX_ITER = settings["maxIters"] # 300 is good for small datasets

    # Train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    new_model_path = os.path.join(model_path, "model_999.pth")
    os.replace(os.path.join(output_path, "model_final.pth"), new_model_path)

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
