# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import os
import argparse
import torch
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T

import Pyro4

from droidlet.perception.robot.semantic_mapper.constants import (
    coco_categories_mapping,
    coco_categories,
    frame_color_palette,
)

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True


@Pyro4.expose
class SemanticPredMaskRCNN(object):
    def __init__(self, robot=None, sem_pred_prob_thr=0.8, sem_gpu_id=-1, visualize=False):
        self.visualize = visualize
        self.num_sem_categories = len(coco_categories)
        # the robot is here in case we are in sim, and the robot will return the categories given by sim.
        self.robot = robot
        self.one_hot_encoding = np.eye(self.num_sem_categories)
        self.scene_contains_semantic_annotations = False
        if robot is not None:
            self.scene_contains_semantic_annotations = robot.scene_contains_semantic_annotations()
        if self.scene_contains_semantic_annotations:
            print("Scene contains semantic annotations")
            (
                self.instance_id_to_category_id,
                self.categories_present,
            ) = self.robot.get_instance_id_to_category_id()
        else:
            print("Scene does not contain semantic annotations, loading model")
            self.segmentation_model = ImageSegmentation(sem_pred_prob_thr, sem_gpu_id)

    def get_semantic_frame_vis(self, rgb, semantics):
        """Visualize first-person semantic segmentation frame."""
        width, height = semantics.shape[:2]
        vis_content = semantics
        vis_content[:, :, -1] = 1e-5
        vis_content = vis_content.argmax(-1)
        vis = Image.new("P", (height, width))
        vis.putpalette([int(x * 255.0) for x in frame_color_palette])
        vis.putdata(vis_content.flatten().astype(np.uint8))
        vis = vis.convert("RGB")
        vis = np.array(vis)
        vis = np.where(vis != 255, vis, rgb)
        return vis

    def get_semantics(self, rgb, depth, rotate=False):
        if self.scene_contains_semantic_annotations:
            instance_segmentation = self.robot.get_rgb_depth_segm()[2]
            semantic_segmentation = self.instance_id_to_category_id[instance_segmentation]
            semantics = self.one_hot_encoding[semantic_segmentation]
            semantics_vis = self.get_semantic_frame_vis(rgb, semantics)
        else:
            semantics, semantics_vis = self.get_prediction(rgb)

        if rotate:
            # given RGB and depth are rotated after the point cloud creation,
            # we rotate them back here to align to the point cloud
            depth = np.rot90(depth, k=1, axes=(0, 1))
            semantics = np.rot90(semantics, k=1, axes=(0, 1))

        # apply the same depth filter to semantics as we applied to the point cloud
        semantics = semantics.reshape(-1, self.num_sem_categories)
        valid = (depth > 0).flatten()
        semantics = semantics[valid]

        return semantics, semantics_vis

    def get_prediction(self, img):
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=self.visualize
        )

        semantic_pred = np.zeros((img.shape[0], img.shape[1], self.num_sem_categories))

        for j, class_idx in enumerate(seg_predictions[0]["instances"].pred_classes.cpu().numpy()):
            if class_idx in list(coco_categories_mapping.keys()):
                idx = coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]["instances"].pred_masks[j] * 1.0
                semantic_pred[:, :, idx] += obj_mask.cpu().numpy()

        if self.visualize:
            img = vis_output.get_image()

        return semantic_pred, img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.0] = i + 1
    return c_map


class ImageSegmentation:
    def __init__(self, sem_pred_prob_thr, sem_gpu_id):
        string_args = f"""
            --config-file {os.path.dirname(os.path.abspath(__file__)) + "/mask_rcnn_R_50_FPN_3x.yaml"}
            --input input1.jpeg
            --confidence-threshold {sem_pred_prob_thr}
            --opts MODEL.WEIGHTS
            detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
            """
        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""
        string_args = string_args.split()
        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=False):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return all_predictions, vis_output


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    Compared to using the model directly, this class does the following
    additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    #    parser.add_argument(
    #        "--ip", help="local IP. Default is 192.168.0.0", type=str, default="0.0.0.0"
    #    )
    #    parser.add_argument(
    #        "--robot_ip", help="remote robot ip. Default is 192.168.0.0", type=str, default="0.0.0.0"
    #    )
    parser.add_argument("--robot_name", default="remotelocobot")
    robot_ip = os.getenv("LOCOBOT_IP")
    ip = os.getenv("LOCAL_IP")
    args = parser.parse_args()

    with Pyro4.Daemon(ip) as daemon:
        robot = Pyro4.Proxy("PYRONAME:" + args.robot_name + "@" + robot_ip)
        S = SemanticPredMaskRCNN(robot)
        uri = daemon.register(S)
        with Pyro4.locateNS(host=ip) as ns:
            ns.register("scene_semantics", uri)

        print("scene semantic prediction server is started...")

        def callback():
            time.sleep(0.0)
            return True

        daemon.requestLoop(callback)
