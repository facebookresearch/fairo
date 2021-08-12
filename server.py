import os
import logging
import base64
import cv2
from imantics import Mask, Polygons
import numpy as np
from PIL import Image
from pathlib import Path
import json
from pycococreatortools import pycococreatortools
from sklearn.model_selection import train_test_split
import time

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from droidlet import dashboard
if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
from agents.argument_parser import ArgumentParser
from droidlet.event import sio
from droidlet.perception.robot import LabelPropagate
import agents.locobot.label_prop as LP
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

class OfflineInstance(): 

    def __init__(self): 
        self.init_event_handlers()
    
    def init_event_handlers(self): 
        @sio.on("offline_label_propagation")
        def offline_label_propagation(sid, postData): 
            """
            postData: 
                filepath: where to access rgb/depth/pose data on disk
                srcFrame: source frame id
                curFrame: current frame id (label prop can go in either dir)
                objects: array of objects
                
            Returns a dictionary mapping objects ids to the objects with new 
            masks. Different from online LP because offline uses data from disk. 
            """

            # Get rgb image and depth map
            rgb_path = os.path.join(postData["filepath"], "rgb")
            depth_path = os.path.join(postData["filepath"], "depth")
            cur_num_zeros = 5 - len(str(postData["curFrame"]))
            cur_file_num = "".join(["0" for _ in range(cur_num_zeros)]) + str(postData["curFrame"])
            src_num_zeros = 5 - len(str(postData["srcFrame"]))
            src_file_num = "".join(["0" for _ in range(src_num_zeros)]) + str(postData["srcFrame"])
            
            rgb_filename = os.path.join(rgb_path, cur_file_num + ".jpg")
            src_img = cv2.imread(rgb_filename)
            height, width, _ = src_img.shape

            depth_filename = os.path.join(depth_path, cur_file_num + ".npy")
            cur_depth = np.load(depth_filename)
            src_depth_filename = os.path.join(depth_path, src_file_num + ".npy")
            src_depth = np.load(src_depth_filename)

            # Labels map
            src_label = LP.mask_to_map(postData["objects"], height, width)

            # Attach base pose data
            pose_filepath = os.path.join(postData["filepath"], "data.json")
            with open(pose_filepath, "rt") as file: 
                pose_dict = json.load(file)
            src_pose = pose_dict[str(postData["srcFrame"])]
            cur_pose = pose_dict[str(postData["curFrame"])]
            
            LabelProp = LabelPropagate()
            res_labels = LabelProp(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)

            # Convert mask maps to mask points
            objects = LP.labels_to_objects(res_labels, postData["objects"])

            # Returns an array of objects with updated masks
            sio.emit("labelPropagationReturn", objects)
        
        @sio.on("offline_save_rgb_seg")
        def offline_save_rgb_seg(sid, postData): 
            """
            postData: 
                filepath: where to access rgb data on disk
                frameId: which file to access on disk
                outputId: counter to help name output files
                categories: array starting with null of categories saved in dashboard
                objects: array of objects with masks and labels
                finalFrame: boolean for when to save annotations to COCO format

            Saves rgb image into annotation_data/rgb and creates a segmentation 
            map to be saved in annotation_data/seg. Also saves all annotations 
            to COCO format if needed. 
            """

            # Get rgb image
            rgb_path = os.path.join(postData["filepath"], "rgb")
            num_zeros = 5 - len(str(postData["frameId"]))
            file_num = "".join(["0" for _ in range(num_zeros)]) + str(postData["frameId"])
            rgb_filename = os.path.join(rgb_path, file_num + ".jpg")
            rgb = cv2.imread(rgb_filename)
            height, width, _ = rgb.shape

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
            np.save("annotation_data/seg/{:05d}.npy".format(postData["outputId"]), display_map)
            im = Image.fromarray(rgb)
            im.save("annotation_data/rgb/{:05d}.jpg".format(postData["outputId"]))

            if postData["finalFrame"]: 
                LP.save_annotations(postData["categories"])

        @sio.on("save_categories_properties")
        def save_categories_properties(sid, categories, properties): 
            LP.save_categories_properties(categories, properties)

        @sio.on("retrain_detector")
        def retrain_detector(sid, settings={}): 
            inference_json = LP.retrain_detector(settings)
            sio.emit("annotationRetrain", inference_json)
        
        # Send first frame in folder
        @sio.on("start_offline_dashboard")
        def start_offline_dashboard(sid, filepath): 
            num_files = len(os.listdir(os.path.join(filepath, "rgb")))
            sio.emit("handleMaxFrames", num_files - 1) # Minus 1 because filenames start at 00000

        # Send first frame in folder
        @sio.on("get_offline_frame")
        def get_offline_frame(sid, data): 
            rgb, depth = LP.get_offline_frame(data)
            sio.emit("rgb", rgb)
            sio.emit("depth", depth)


if __name__ == "__main__":
    
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Locobot", base_path)
    opts = parser.parse()

    logging.basicConfig(level=opts.log_level.upper())
    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    oi = OfflineInstance()

    while True: 
        time.sleep(1)
