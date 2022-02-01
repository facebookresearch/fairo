import numpy as np
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import random
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image
from pycococreatortools import pycococreatortools
from IPython import embed
from tqdm import tqdm
from IPython.core.display import display, HTML
import os
import shutil
import glob 
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from common_utils import (
    is_annot_validfn_class, 
    is_annot_validfn_inst,
    instance_ids,
    class_labels,
)

semantic_json_root = '/checkpoint/apratik/ActiveVision/active_vision/info_semantic'

class CocoCreator:
    # Assumes root_data_dir has both te GT and propagated segmentation labels
    def __init__(self, root_data_dir, semantic_json_root, is_valid_annot_fn, setting):
        self.rdd = root_data_dir
        self.sjr = semantic_json_root
        self.segm_dir = os.path.join(root_data_dir, 'seg')
        self.is_valid_annot_fn = is_valid_annot_fn
        self.create_metadata = self.create_metadata_inst if setting == 'instance' else self.create_metadata_class
        
    def create_coco(self, scene, coco_file_name, pct):
        hsd = self.load_semantic_json(scene)
        self.create_metadata(hsd)
        self.create_annos(hsd, scene, pct)
        self.save_json(coco_file_name)
        self.save_visual_dataset(coco_file_name, scene)
    
    def save_visual_dataset(self, coco_file_name, scene):
        DatasetCatalog.clear()
        MetadataCatalog.clear()

        register_coco_instances('foobar', {}, coco_file_name, os.path.join(self.rdd, 'rgb'))
        MetadataCatalog.get('foobar')
        dataset_dicts = DatasetCatalog.get('foobar')
        
        save_dir = os.path.join(self.rdd, 'coco_visuals')
        print(f'save_dir {save_dir}, coco_file_name {coco_file_name}')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            x = d['file_name'].split('/')[-1]
            print(f"filename {d['file_name'], x}, visual_file {os.path.join(save_dir, x)}")
            visualizer = Visualizer(img[:,:,::-1], metadata=MetadataCatalog.get('foobar'), scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            img = vis.get_image()
            cv2.imwrite(os.path.join(save_dir, x), img[:,:,::-1])
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.show()
        
    def save_json(self, coco_file_name):
        coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": self.IMAGES,
            "annotations": self.ANNOTATIONS,
        }
        print(f'self.CATS {self.CATEGORIES}')
        print(f"Dumping {len(coco_output['annotations'])} annotations to {coco_file_name}")
        with open(coco_file_name, "w") as output_json:
            json.dump(coco_output, output_json)
        
    def create_metadata_class(self, hsd):
        self.INFO = {}
        self.LICENSES = [{}]
        self.CATEGORIES = []
        self.IMAGES = []
        self.ANNOTATIONS = []
        
        self.label_id_dict = {}
        self.new_old_id = {}
        idc = 1
        for obj_cls in hsd["classes"]:
            if obj_cls["name"] in self.labels:
                self.CATEGORIES.append({"id": idc, "name": obj_cls["name"], "supercategory": "shape"})
                self.label_id_dict[obj_cls["id"]] = obj_cls["name"]
                self.new_old_id[obj_cls['id']] = idc
                idc += 1

    def create_metadata_inst(self, hsd):
        self.INFO = {}
        self.LICENSES = [{}]
        self.CATEGORIES = []
        self.IMAGES = []
        self.ANNOTATIONS = []
        self.class_id_label = {}
        for x in hsd['classes']:
            self.class_id_label[x['id']] = x['name']
        self.label_id_dict = {}
        self.new_old_id = {}
        idc = 1
        for i in instance_ids:
            instance_label = str(i) + '_' + self.class_id_label[hsd['id_to_label'][i]]
            print(f'instance_label {instance_label}')
            self.CATEGORIES.append({"id": idc, "name": instance_label, "supercategory": "shape"})
            self.label_id_dict[i] = instance_label # name the classes as instance_id + class 
            self.new_old_id[i] = idc
            idc += 1
    
    def create_annos(self, hsd, scene, pct):
        coco_img_id = -1
        count = 0
        segm_dir = self.segm_dir
        print(f"Scene {scene}, seg dir {segm_dir}")       
        img_dir = os.path.join(self.rdd, 'rgb')
        fs = self.get_segm_files(segm_dir, pct)
        print(f"Creating COCO annotations for {len(fs)} images \n img_dir {img_dir}")
        
        for f in tqdm(fs):
            image_id = int(f.split('.')[0])
            try:
                prop_path = os.path.join(segm_dir, "{:05d}.npy".format(image_id))
                annot = np.load(prop_path).astype(np.uint32)
            except Exception as e:
                print(e)
                continue

            img_filename = "{:05d}.jpg".format(image_id)            
            img = Image.open(os.path.join(img_dir, img_filename))

            # COCO ID and names
            coco_img_id += 1

            image_info = pycococreatortools.create_image_info(
                coco_img_id, os.path.basename(img_filename), img.size
            )

            self.IMAGES.append(image_info)
            
            for i in np.sort(np.unique(annot.reshape(-1), axis=0)):
                if self.is_valid_annot_fn(i):
                    try:
                        if hsd["id_to_label"][i] < 1:# or hsd["id_to_label"][i] not in self.label_id_dict:
                            continue
                        category_info = {"id": self.new_old_id[i], "is_crowd": False}
                    except:
                        continue

                    binary_mask = (annot == i).astype(np.uint32)

                    annotation_info = pycococreatortools.create_annotation_info(
                        count, coco_img_id, category_info, binary_mask, img.size, tolerance=2
                    )
                    if annotation_info is not None:
                        self.ANNOTATIONS.append(annotation_info)
                        count += 1
        
    def load_semantic_json(self, scene):
        replica_root = '/datasets01/replica/061819/18_scenes'
        habitat_semantic_json = os.path.join(replica_root, scene, 'habitat', 'info_semantic.json')
        with open(habitat_semantic_json, "r") as f:
            hsd = json.load(f)
        if hsd is None:
            print("Semantic json not found!")
        return hsd
        
    def get_segm_files(self, segm_dir, pct):
        cs = [os.path.basename(x) for x in glob.glob(os.path.join(segm_dir, '*.npy'))]
        cs.sort()
        frq = 1/pct
        fs = []
        for x in range(0, len(cs), int(frq)):
            fs.append(cs[x])
        return fs 

def get_valid_annot_fn(root_data_dir):
    if 'instance' in root_data_dir:
        return is_annot_validfn_inst, 'instance'
    if 'class' in root_data_dir:
        return is_annot_validfn_class, 'class'

def run_coco(root_data_dir):
    valid_annot_fn, setting = get_valid_annot_fn(root_data_dir)
    cbase = CocoCreator(root_data_dir, semantic_json_root, valid_annot_fn, setting)
    cbase.create_coco(
        scene='apartment_0', 
        coco_file_name=os.path.join(root_data_dir, 'coco_train.json'),
        pct=1,
    )