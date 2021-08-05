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
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

class OfflineInstance(): 

    def __init__(self): 
        self.init_event_handlers()
    
    def init_event_handlers(self): 
        @sio.on("offline_label_propagation")
        def offline_label_propagation(sid, postData): 
                        
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

            # Convert mask points to mask maps then combine them
            src_label = np.zeros((height, width)).astype(int)
            for n, o in enumerate(postData["objects"]): 
                poly = Polygons(o["mask"])
                bitmap = poly.mask(height, width)
                for i in range(height): 
                    for j in range(width): 
                        if bitmap[i][j]: 
                            src_label[i][j] = n + 1

            # Attach base pose data
            pose_filepath = os.path.join(postData["filepath"], "data.json")
            with open(pose_filepath, "rt") as file: 
                pose_dict = json.load(file)
            src_pose = pose_dict[str(postData["srcFrame"])]
            cur_pose = pose_dict[str(postData["curFrame"])]
            
            LP = LabelPropagate()
            res_labels = LP(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)
            print("label prop output shape:", res_labels.shape)

            # Convert mask maps to mask points
            objects = {}
            for i_float in np.unique(res_labels): 
                i = int(i_float)
                if i == 0: 
                    continue
                objects[i-1] = postData["objects"][i-1] # Do this in the for loop cause some objects aren't returned
                mask_points_nd = Mask(np.where(res_labels == i, 1, 0)).polygons().points
                mask_points = list(map(lambda x: x.tolist(), mask_points_nd))
                objects[i-1]["mask"] = mask_points
                objects[i-1]["type"] = "annotate"

            # Returns an array of objects with updated masks
            sio.emit("labelPropagationReturn", objects)
        
        @sio.on("offline_save_rgb_seg")
        def offline_save_rgb_seg(sid, postData): 

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
                save_annotations(postData["categories"])

        # Adapted from coco_creator.ipynb
        # Use in offline_save_rgb_seg socket event
        def save_annotations(categories): 
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

        @sio.on("save_categories_properties")
        def save_categories_properties(sid, categories, properties): 

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

        @sio.on("retrain_detector")
        def retrain_detector(sid, settings={}): 
            
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

            if train_data in DatasetCatalog.list(): 
                DatasetCatalog.remove(train_data)
            register_coco_instances(train_data, {}, train_path, image_dir)
            if test_data in DatasetCatalog.list(): 
                DatasetCatalog.remove(test_data)
            register_coco_instances(test_data, {}, train_path, image_dir)

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
            sio.emit("annotationRetrain", inference_json)
        
        # Send first frame in folder
        @sio.on("start_offline_dashboard")
        def start_offline_dashboard(sid, filepath): 
            num_files = len(os.listdir(os.path.join(filepath, "rgb")))
            sio.emit("handleMaxFrames", num_files - 1) # Minus 1 because filenames start at 00000

        # Send first frame in folder
        @sio.on("get_offline_frame")
        def get_offline_frame(sid, data): 
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

            sio.emit("rgb", base64.b64encode(rgb_data).decode("utf-8"))
            sio.emit("depth", {
                "depthImg": base64.b64encode(depth_img_data).decode("utf-8"),
                "depthMax": str(np.max(depth)),
                "depthMin": str(np.min(depth)),
            })


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
