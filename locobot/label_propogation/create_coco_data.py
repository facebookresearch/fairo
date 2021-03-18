# helpfule for creating coco data based on the binary annotation images
from PIL import Image
import os
import pycococreatortools
import numpy as np
from IPython import embed
import json
from pycocotools.coco import COCO


# Input ##
img_root_path = (
    "/checkpoint/dhirajgandhi/active_vision/replica_random_exploration_data/apartment_0/rgb"
)
img_annot_root_path = (
    "/checkpoint/dhirajgandhi/active_vision/replica_random_exploration_data/apartment_0/seg"
)
habitat_semantic_json = "info_semantic.json"
train_json = "/checkpoint/apratik/ActiveVision/train.json"
exclude_categories = []
# exclude_categories = ["floor", "ceiling", "wall"]
###

# load train data to know for which images to do label propogation
train_json_data = COCO(train_json)

with open(habitat_semantic_json, "r") as f:
    habitat_semantic_data = json.load(f)

INFO = {}

LICENSES = [{}]

# create categories out of it
CATEGORIES = []
exclude_categories_id = []
for obj_cls in habitat_semantic_data["classes"]:
    if obj_cls["name"] in exclude_categories:
        exclude_categories_id.append(obj_cls["id"])
    else:
        CATEGORIES.append(
            {"id": obj_cls["id"], "name": obj_cls["name"], "supercategory": "shape"}
        )

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": [],
}
annot_id = 0
image_id = 0
skip = 1
propogation = 0
#for train_img_id in train_json_data.getImgIds():
for train_img_id in train_json_data.getImgIds():
    for img_indx in range(
        max(train_img_id - propogation, 0), train_img_id + propogation + 1, skip
    ):
        # load the annotation file
        try:
            annot = np.load(os.path.join(img_annot_root_path, "{:05d}.npy".format(img_indx)))
        except:
            continue

        img_filename = "{:05d}.jpg".format(img_indx)
        img = Image.open(os.path.join(img_root_path, img_filename))
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(img_filename), img.size
        )
        image_id += 1
        coco_output["images"].append(image_info)
        print("image_indx = {}".format(img_indx))

        # for each annotation add to coco format
        for i in np.unique(annot.reshape(-1), axis=0):
            try:
                if habitat_semantic_data["id_to_label"][i] in exclude_categories_id:
                    continue
                category_info = {"id": habitat_semantic_data["id_to_label"][i], "is_crowd": False}
            except:
                print("label value doesnt exist")
                continue
            if category_info["id"] < 0:
                continue
            binary_mask = (annot == i).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                annot_id, image_id, category_info, binary_mask, img.size, tolerance=2
            )
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                annot_id += 1

with open("train_gt.json".format(skip, propogation), "w") as output_json:
    json.dump(coco_output, output_json)
