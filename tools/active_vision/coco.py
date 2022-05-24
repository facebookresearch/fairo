import numpy as np
import sys
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
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode


class CocoCreator:
    # Assumes root_data_dir has both te GT and propagated segmentation labels
    def __init__(self, root_data_dir):
        self.rdd = root_data_dir
        self.segm_dir = os.path.join(root_data_dir, 'seg')
        self.visuals_dir = 'coco_visuals'
        
    def create_coco(self, scene, coco_file_name, pct, use_gt):
        if use_gt:
            self.segm_dir = os.path.join(self.rdd, 'seg_gt')
            self.visuals_dir = 'coco_visuals_gt'
        self.create_metadata()
        self.create_annos(scene, pct)
        self.save_json(coco_file_name)
        self.save_visual_dataset(coco_file_name, scene)
    
    def save_visual_dataset(self, coco_file_name, scene):
        DatasetCatalog.clear()
        MetadataCatalog.clear()

        register_coco_instances('foobar', {}, coco_file_name, os.path.join(self.rdd, 'rgb'))
        MetadataCatalog.get('foobar')
        dataset_dicts = DatasetCatalog.get('foobar')
        
        save_dir = os.path.join(self.rdd, self.visuals_dir)
        print(f'save_dir {save_dir}, coco_file_name {coco_file_name}')
        # if os.path.exists(save_dir):
        #     shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            x = d['file_name'].split('/')[-1]
            # print(f"filename {d['file_name'], x}, visual_file {os.path.join(save_dir, x)}")
            visualizer = Visualizer(img[:,:,::-1], metadata=MetadataCatalog.get('foobar'), scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            img = vis.get_image()
            cv2.imwrite(os.path.join(save_dir, x), img[:,:,::-1])
            # plt.figure(figsize=(4, 4))
            # plt.imshow(img)
            # plt.show()
        
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
        
    def create_metadata(self):
        self.INFO = {}
        self.LICENSES = [{}]
        self.CATEGORIES = []
        self.IMAGES = []
        self.ANNOTATIONS = []
        self.label_to_id = {}
        idc = 1
        for i, label in id_to_label.items():
            instance_label = str(i) + '_' + label
            print(f'instance_label {instance_label}')
            self.CATEGORIES.append({"id": idc, "name": instance_label, "supercategory": "shape"})
            self.label_to_id[i] = idc
            idc += 1
    
    def create_annos(self, scene, pct):
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
                if i in id_to_label:
                    try:
                        category_info = {"id": self.label_to_id[i], "is_crowd": False}
                        # print(f'category_info {category_info}')
                    except Exception as ex:
                        print(f'exception {ex}')
                        continue

                    binary_mask = (annot == i).astype(np.uint32)

                    annotation_info = pycococreatortools.create_annotation_info(
                        count, coco_img_id, category_info, binary_mask, img.size, tolerance=2
                    )
                    if annotation_info is not None:
                        self.ANNOTATIONS.append(annotation_info)
                        count += 1
        
    def get_segm_files(self, segm_dir, pct):
        cs = [os.path.basename(x) for x in glob.glob(os.path.join(segm_dir, '*.npy'))]
        print(f'get_segm_files {segm_dir}, pct {pct}, total_files {len(cs)}')
        cs.sort()
        frq = 1/pct
        fs = []
        for x in range(0, len(cs), int(frq)):
            fs.append(cs[x])
        return fs 

id_to_label = {
    87: 'plant',
#     97: 'jacket',
    103: 'hamper',
    # 106: 'sidetable',
    140: 'armchair',
}

def run_coco(root_data_dir):
    cbase = CocoCreator(root_data_dir)
    cbase.create_coco(
        scene='soumith_apt', 
        coco_file_name=os.path.join(root_data_dir, 'coco_train.json'),
        pct=1,
        use_gt=False,
    )
    
