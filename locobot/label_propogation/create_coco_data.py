# helpfule for creating coco data based on the binary annotation images
from PIL import Image
import os
import pycococreatortools
import numpy as np
import cv2
from IPython import embed
import json

# Input ##
img_root_path = "/checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/rgb"
img_annot_root_path = "/checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/seg"
habitat_semantic_json = "info_semantic.json"
img_range = [6500, 6600]
###

with open(habitat_semantic_json, "r") as f:
    habitat_semantic_data = json.load(f)

INFO = {}

LICENSES = [{}]

# create categories out of it
CATEGORIES = []
for obj_cls in habitat_semantic_data["classes"]:
    CATEGORIES.append({"id": obj_cls["id"], "name": obj_cls["name"], "supercategory": "shape"})

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": [],
}

count = 0
for image_id, img_indx in enumerate(range(img_range[0], img_range[1])):
    img_filename = "{:05d}.jpg".format(img_indx)
    img = Image.open(os.path.join(img_root_path, img_filename))
    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(img_filename), img.size
    )
    coco_output["images"].append(image_info)
    print("image_indx = {}".format(img_indx))

    # load the annotation file
    annot = np.load(os.path.join(img_annot_root_path, "{:05d}.npy".format(img_indx)))

    # for each annotation add to coco format
    for i in np.unique(annot.reshape(-1), axis=0):

        category_info = {"id": habitat_semantic_data["id_to_label"][i], "is_crowd": False}
        binary_mask = (annot == i).astype(np.uint8)

        annotation_info = pycococreatortools.create_annotation_info(
            count, image_id, category_info, binary_mask, img.size, tolerance=2
        )
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
            count += 1

with open("habitat_sem_annot.json", "w") as output_json:
    json.dump(coco_output, output_json)
