"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import subprocess
import time
import signal
import random
import logging
import faulthandler
from multiprocessing import set_start_method
import base64
import cv2
from imantics import Mask, Polygons
import numpy as np
from PIL import Image
from pathlib import Path
import json
from pycococreatortools import pycococreatortools
from sklearn.model_selection import train_test_split
import pickle

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
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from droidlet.base_util import to_player_struct, Pos, Look, Player
from droidlet.memory.memory_nodes import PlayerNode
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from agents.loco_mc_agent import LocoMCAgent
from agents.argument_parser import ArgumentParser
from droidlet.memory.robot.loco_memory import LocoAgentMemory, DetectedObjectNode
from droidlet.perception.robot import Perception
from droidlet.perception.semantic_parsing.utils.interaction_logger import InteractionLogger
from self_perception import SelfPerception
from droidlet.interpreter.robot import (
    dance, 
    default_behaviors,
    LocoGetMemoryHandler, 
    PutMemoryHandler, 
    LocoInterpreter,
)
from droidlet.dialog.robot import LocoBotCapabilities
import droidlet.lowlevel.locobot.rotation as rotation
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
from droidlet.event import sio
from droidlet.perception.robot import LabelPropagate

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()


class LocobotAgent(LocoMCAgent):
    """Implements an instantiation of the LocoMCAgent on a Locobot. It starts
    off the agent processes including launching the dashboard.

    Args:
        opts (argparse.Namespace): opts returned by the ArgumentParser with defaults set
            that you can override.
        name (string, optional): a name for your agent (default: Locobot)

    Example:
        >>> python locobot_agent.py --backend 'locobot'
    """

    coordinate_transforms = rotation

    def __init__(self, opts, name="Locobot"):
        super(LocobotAgent, self).__init__(opts)
        logging.info("LocobotAgent.__init__ started")
        self.opts = opts
        self.entityId = 0
        self.no_default_behavior = opts.no_default_behavior
        self.last_chat_time = -1000000000000
        self.name = name
        self.player = Player(100, name, Pos(0, 0, 0), Look(0, 0))
        self.pos = Pos(0, 0, 0)
        self.uncaught_error_count = 0
        self.last_task_memid = None
        self.point_targets = []
        self.init_event_handlers()
        # list of (prob, default function) pairs
        self.visible_defaults = [(1.0, default_behaviors.explore)]
        self.interaction_logger = InteractionLogger()

    def init_event_handlers(self):
        super().init_event_handlers()

        @sio.on("movement command")
        def test_command(sid, commands):
            movement = [0.0, 0.0, 0.0]
            for command in commands:
                if command == "MOVE_FORWARD":
                    movement[0] += 0.1
                    print("action: FORWARD")
                elif command == "MOVE_BACKWARD":
                    movement[0] -= 0.1
                    print("action: BACKWARD")
                elif command == "MOVE_LEFT":
                    movement[2] += 0.3
                    print("action: LEFT")
                elif command == "MOVE_RIGHT":
                    movement[2] -= 0.3
                    print("action: RIGHT")
                elif command == "PAN_LEFT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() + 0.08)
                elif command == "PAN_RIGHT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() - 0.08)
                elif command == "TILT_UP":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() - 0.08)
                elif command == "TILT_DOWN":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() + 0.08)
            self.mover.move_relative([movement])

        @sio.on("shutdown")
        def _shutdown(sid, data):
            self.shutdown()

        @sio.on("get_memory_objects")
        def objects_in_memory(sid):
            objects = DetectedObjectNode.get_all(self.memory)
            for o in objects:
                del o["feature_repr"] # pickling optimization
            self.dashboard_memory["objects"] = objects
            sio.emit("updateState", {"memory": self.dashboard_memory})
        
        @sio.on("interaction data")
        def log_interaction_data(sid, interactionData):
            self.interaction_logger.logInteraction(interactionData)

        @sio.on("label_propagation")
        def label_propagation(sid, postData): 
                        
            # Decode rgb map
            rgb_bytes = base64.b64decode(postData["prevRgbImg"])
            rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
            rgb_bgr = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
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
                depth_imgs.append(depth_scaled)
            src_depth = np.array(depth_imgs[0])
            cur_depth = np.array(depth_imgs[1])

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
            pose = postData["prevBasePose"]
            src_pose = np.array([pose["x"], pose["y"], pose["yaw"]])
            pose = postData["basePose"]
            cur_pose = np.array([pose["x"], pose["y"], pose["yaw"]])
            
            LP = LabelPropagate()
            res_labels = LP(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)

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
        
        @sio.on("save_rgb_seg")
        def save_rgb_seg(sid, postData): 

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

            if "callback" in postData and postData["callback"]: 
                sio.emit("saveRgbSegCallback")

        @sio.on("save_annotations")
        def save_annotations(sid, categories): 
            if len(categories) == 0: 
                print("Error in saving annotations: Categories need to not be null. \
                    You cannot just use the rgb/ and seg/ folders to create the \
                        COCO json -- categories do not persist in memory")
                return

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
            }

            count = 0
            for x in fs:
                image_id = int(x.split(".")[0])
                # load the annotation file
                try:
                    prop_path = os.path.join(seg_dir, "{:05d}.npy".format(image_id))
                    annot = np.load(prop_path).astype(np.uint8)
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
            dataset_name = "annotation_coco"
            image_dir = base_path + "rgb/"
            train_data = dataset_name + "_train"
            test_data = dataset_name + "_test"

            if train_data in DatasetCatalog.list(): 
                DatasetCatalog.remove(train_data)
            if train_data in MetadataCatalog.list(): 
                MetadataCatalog.remove(train_data)
            register_coco_instances(train_data, {}, train_path, image_dir)
            if test_data in DatasetCatalog.list(): 
                DatasetCatalog.remove(test_data)
            if test_data in MetadataCatalog.list(): 
                MetadataCatalog.remove(test_data)
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

        @sio.on("switch_detector")
        def switch_detector(sid): 
            model_path = "annotation_data/model"
            detector_weights = "model_999.pth"
            properties_file = "props.json"
            things_file = "things.json"

            files = os.listdir(model_path)
            if detector_weights not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return
            if properties_file not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return
            if things_file not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return

            print("switching to", model_path)
            self.perception_modules["vision"] = Perception(model_path, default_keypoints_path=True)


    def init_memory(self):
        """Instantiates memory for the agent.

        Uses the DB_FILE environment variable to write the memory to a
        file or saves it in-memory otherwise.
        """
        self.memory = LocoAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path=None,
            coordinate_transforms=self.coordinate_transforms,
        )
        dance.add_default_dances(self.memory)
        logging.info("Initialized agent memory")

    def init_perception(self):
        """Instantiates all perceptual modules.

        Each perceptual module should have a perceive method that is
        called by the base agent event loop.
        """
        self.chat_parser = NSPQuerier(self.opts)
        if not hasattr(self, "perception_modules"):
            self.perception_modules = {}
        self.perception_modules["self"] = SelfPerception(self)
        self.perception_modules["vision"] = Perception(self.opts.perception_model_dir)

    def perceive(self, force=False):
        self.perception_modules["self"].perceive(force=force)


        rgb_depth = self.mover.get_rgb_depth()
        xyz = self.mover.get_base_pos_in_canonical_coords()
        x, y, yaw = xyz
        sio.emit("map", {
            "x": x,
            "y": y,
            "yaw": yaw,
            "map": self.mover.get_obstacles_in_canonical_coords()
        })

        previous_objects = DetectedObjectNode.get_all(self.memory)
        new_state = self.perception_modules["vision"].perceive(rgb_depth,
                                                               xyz,
                                                               previous_objects,
                                                               force=force)
        if new_state is not None:
            new_objects, updated_objects = new_state
            for obj in new_objects:
                obj.save_to_memory(self.memory)
            for obj in updated_objects:
                obj.save_to_memory(self.memory, update=True)

    def init_controller(self):
        """Instantiates controllers - the components that convert a text chat to task(s)."""
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = LocoBotCapabilities
        dialogue_object_classes["interpreter"] = LocoInterpreter
        dialogue_object_classes["get_memory"] = LocoGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.dialogue_manager = DialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
        )

    def init_physical_interfaces(self):
        """Instantiates the interface to physically move the robot."""
        self.mover = LoCoBotMover(ip=self.opts.ip, backend=self.opts.backend)

    def get_player_struct_by_name(self, speaker_name):
        p = self.memory.get_player_by_name(speaker_name)
        if p:
            return p.get_struct()
        else:
            return None

    def get_other_players(self):
        return [self.player]

    def get_incoming_chats(self):
        all_chats = []
        speaker_name = "dashboard"
        if self.dashboard_chat is not None:
            if not self.memory.get_player_by_name(speaker_name):
                PlayerNode.create(
                    self.memory,
                    to_player_struct((None, None, None), None, None, None, speaker_name),
                )
            all_chats.append(self.dashboard_chat)
            self.dashboard_chat = None
        return all_chats

    # # FIXME!!!!
    def send_chat(self, chat: str):
        logging.info("Sending chat: {}".format(chat))
        # Send the socket event to show this reply on dashboard
        sio.emit("showAssistantReply", {"agent_reply": "Agent: {}".format(chat)})
        self.memory.add_chat(self.memory.self_memid, chat)
        # actually send the chat, FIXME FOR HACKATHON
        # return self._cpp_send_chat(chat)

    def step(self):
        super().step()
        time.sleep(0)

    def task_step(self, sleep_time=0.0):
        super().task_step(sleep_time=sleep_time)

    def shutdown(self):
        self._shutdown = True
        try:
            self.perception_modules["vision"].vprocess_shutdown.set()
        except:
            """
            the try/except is there in the event that
            self.perception_modules["vision"] has either:
            1. not been fully started yet
            2. already crashed / shutdown due to other effects
            """
            pass
        time.sleep(5) # let the other threads die
        os._exit(0) # TODO: remove and figure out why multiprocess sometimes hangs on exit


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

    # Check that models and datasets are up to date
    if not opts.dev:
        rc = subprocess.call([opts.verify_hash_script_path, "locobot"])

    set_start_method("spawn", force=True)

    sa = LocobotAgent(opts)
    sa.start()
