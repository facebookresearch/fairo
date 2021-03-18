# helpfule for creating coco data based on the binary annotation images
from PIL import Image
import os
import pycococreatortools
import numpy as np
from IPython import embed
import json
from pycocotools.coco import COCO
import glob
import argparse


def main(
    img_root_path: str,
    img_annot_root_path: str,
    train_json: str,
    exclude_categories: list,
    out_file: str,
    habitat_semantic_json="info_semantic.json",
):
    """[summary]

    Args:
        img_root_path (str): [path to rgb image folder]
        img_annot_root_path (str): [path where annottion data is being sored]
        train_json (str): [coco training json file to know for which images prpogated label need to be used]
        exclude_categories (list): [categories that need to be excluded]
        out_file (str): [json file name to store output]
        habitat_semantic_json (str, optional): [habitat json file describing is to label]. Defaults to "info_semantic.json".
    """
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
    count = 0
    # for testing
    img_id_set = set()

    for i in range(len(glob.glob(os.path.join(img_root_path, "*.jpg")))):
        if i not in train_json_data.getImgIds():
            img_id_set.add(i)

    for image_id, img_indx in enumerate(img_id_set):
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
                count, image_id, category_info, binary_mask, img.size, tolerance=2
            )
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                count += 1

    with open(out_file, "w") as output_json:
        json.dump(coco_output, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for creating COCO dataset")
    parser.add_argument(
        "--img_root_path",
        help="Path to image folder",
        type=str,
        default="/checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/rgb",
    )
    parser.add_argument(
        "--img_annot_root_path",
        help="Path to annotation folder",
        type=str,
        default="/checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/pred_label",
    )
    parser.add_argument(
        "--habitat_semantic_json",
        help="json file having information of id to category",
        type=str,
        default="info_semantic.json",
    )
    parser.add_argument(
        "--train_json",
        help="coco json file used for image ids",
        type=str,
        default="/checkpoint/apratik/ActiveVision/train.json",
    )
    parser.add_argument(
        "--out_file", help="json file to store output", type=str, default="coco.json"
    )
    parser.add_argument(
        "--exclude_categories", help="categories to exclude", type=str, default="", nargs="+"
    )
    args = parser.parse_args()
    if type(args.exclude_categories) is str:
        args.exclude_categories = []
    main(
        img_root_path=args.img_root_path,
        img_annot_root_path=args.img_annot_root_path,
        train_json=args.train_json,
        exclude_categories=args.exclude_categories,
        out_file=args.out_file,
        habitat_semantic_json=args.habitat_semantic_json,
    )
