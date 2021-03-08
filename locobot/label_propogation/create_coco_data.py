# helpfule for creating coco data based on the binary annotation images
from PIL import Image
import os
import pycococreatortools
import numpy as np
import cv2
from IPython import embed
import json

INFO = {
}

LICENSES = [
    {
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'sofa',
        'supercategory': 'shape',
    },
    {
        'id': 1,
        'name': 'armchair',
        'supercategory': 'shape',
    },
]

img_root_path = '/checkpoint/dhirajgandhi/active_vision/habitat_data/rgb'
img_annot_root_path = '/checkpoint/dhirajgandhi/active_vision/habitat_data/pred_label'
img_range = [6500, 7500]
coco_output = {"info": INFO, "licenses": LICENSES, "categories": CATEGORIES, "images":[], "annotations":[]}

count = 0
for image_id, img_indx in enumerate(range(img_range[0], img_range[1])):
    img_filename = '{:05d}.jpg'.format(img_indx)
    img = Image.open(os.path.join(img_root_path, img_filename))
    image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(img_filename), img.size)
    coco_output["images"].append(image_info)
    print("image_indx = {}".format(img_indx))
    # for this file search for each annotation
    annotation_files = {}
    # check for id 1,2
    for i in range(2):
        tmp_file = os.path.join(img_annot_root_path, '{:05d}_{}.png'.format(img_indx, i))
        print("tmp_file = {}".format(tmp_file))
        if os.path.isfile(tmp_file):
            annotation_files[i] = tmp_file
            category_info = {'id': i, 'is_crowd': False}
            binary_mask = cv2.imread(tmp_file)
            binary_mask = (binary_mask[:,:,0] == 255).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(count, image_id, category_info, binary_mask,img.size, tolerance=2)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                count += 1

with open("habitat_annot_coco.json", "w") as output_json:
    json.dump(coco_output, output_json)