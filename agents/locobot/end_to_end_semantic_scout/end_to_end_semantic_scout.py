import numpy as np
import torch
import os
import time
from PIL import Image
import cv2
import skimage.morphology
import matplotlib.pyplot as plt
import shutil
import json

from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from .src import POLICY_CLASSES
from .src.default import get_config
from .src.models.common import batch_obs
from .src.models.rednet import load_rednet
from .constants import (
    coco_categories,
    coco_id_to_goal_id,
    expected_categories_to_coco_categories,
    frame_color_palette,
)
from .segmentation.detectron2_segmentation import Detectron2Segmentation
# from .segmentation.mmdetection_segmentation import MMDetectionSegmentation
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
from droidlet.lowlevel.pyro_utils import safe_call


class RLSegFTAgent(Agent):
    def __init__(self, config: Config):
        if not config.MODEL_PATH:
            raise Exception("Model checkpoint wasn't provided, quitting.")
        if config.TORCH_GPU_ID >= 0:
            self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        else:
            self.device = torch.device("cpu")

        self.color_palette = [int(x * 255.0) for x in frame_color_palette]

        ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)["state_dict"]
        ckpt_dict = {k.replace("actor_critic.", ""): v for k, v in ckpt_dict.items()}
        ckpt_dict = {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
        ckpt_dict = {k.replace("model.", ""): v for k, v in ckpt_dict.items()}
        print("ckpt_dict.keys()", ckpt_dict.keys())

        # Config
        self.config = config
        config = self.config.clone()
        self.model_cfg = config.MODEL
        il_cfg = config.IL.BehaviorCloning
        task_cfg = config.TASK_CONFIG.TASK

        # Load spaces (manually)
        spaces = {
            "objectgoal": Box(
                low=0, high=20, shape=(1,), dtype=np.int64  # From matterport dataset
            ),
            "depth": Box(
                low=0,
                high=1,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH,
                    1,
                ),
                dtype=np.float32,
            ),
            "rgb": Box(
                low=0,
                high=255,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
                    3,
                ),
                dtype=np.uint8,
            ),
            "gps": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),  # Spoof for model to be shaped correctly
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float),
        }

        observation_spaces = SpaceDict(spaces)
        if "action_distribution.linear.bias" in ckpt_dict:
            num_acts = ckpt_dict["action_distribution.linear.bias"].size(0)
        action_spaces = Discrete(num_acts)

        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        if is_objectnav:
            additional_sensors = ["gps", "compass"]

        policy_class = POLICY_CLASSES[il_cfg.POLICY.name]
        self.model = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            model_config=self.model_cfg,
            device=self.device,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            additional_sensors=additional_sensors,
        ).to(self.device)

        # self.model.load_state_dict(ckpt_dict, strict=True)
        self.model.load_state_dict(ckpt_dict, strict=False)  # To load model without RL fine-tuning
        self.model.eval()

        self.semantic_predictor = None
        if "mp3d_detector" in config.POLICY:
            if self.model_cfg.USE_SEMANTICS:
                self.semantic_predictor = load_rednet(
                    self.device,
                    ckpt=self.model_cfg.SEMANTIC_ENCODER.rednet_ckpt,
                    resize=True,  # Since we train on half-vision
                    num_classes=self.model_cfg.SEMANTIC_ENCODER.num_classes,
                )
                self.semantic_predictor.eval()
        elif "coco_detector" in config.POLICY:
            self.semantic_predictor = Detectron2Segmentation(
                sem_pred_prob_thr=0.9, sem_gpu_id=config.TORCH_GPU_ID, visualize=True
            )
            # self.semantic_predictor = MMDetectionSegmentation(
            #     sem_pred_prob_thr=0.9, device=self.device, visualize=True
            # )

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1,  # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        self.ep = 0

    def reset(self):
        # We don't reset state because our rnn accounts for masks, and ignore actions because we don't use actions
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1,  # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        self.ep += 1
        logger.info("Episode done: {}".format(self.ep))

    def get_semantic_frame_vis(self, rgb, semantics):
        """Visualize first-person semantic segmentation frame."""
        width, height = semantics.shape
        vis = Image.new("P", (height, width))
        vis.putpalette(self.color_palette)

        # Convert category IDs expected by the policy to Coco
        # category IDs for visualization
        semantics = np.array(
            [
                expected_categories_to_coco_categories.get(idx, coco_categories["no-category"])
                for idx in semantics.flatten()
            ]
        ).astype(np.uint8)

        vis.putdata(semantics.flatten().astype(np.uint8))
        vis = vis.convert("RGB")
        vis = np.array(vis)
        vis = np.where(vis != 255, vis, rgb)
        vis = vis[:, :, [2, 1, 0]]
        return vis

    @torch.no_grad()
    def act(self, observations):

        batch = batch_obs([observations], device=self.device)

        with torch.no_grad():
            if self.semantic_predictor is not None:
                # Replace predictions of segmentation model trained in simulation used
                # to train the policy with detectron2 Mask-RCNN that works much better
                # in the real world (we use only the object goal categories for now)

                if isinstance(self.semantic_predictor, Detectron2Segmentation):
                # if isinstance(self.semantic_predictor, MMDetectionSegmentation):
                    semantic, semantic_vis = self.semantic_predictor.get_prediction(
                        batch["rgb"].cpu().numpy(),
                        batch["depth"].cpu().numpy()
                    )
                    semantic = torch.from_numpy(semantic).to(batch["rgb"].device)
                    semantic_vis = semantic_vis[0]

                else:
                    semantic = self.semantic_predictor(batch["rgb"], batch["depth"])
                    if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                        semantic = semantic - 1
                    semantic_vis = self.get_semantic_frame_vis(
                        batch["rgb"][0].cpu().numpy(), semantic[0].cpu().numpy()
                    )

                batch["semantic"] = semantic

            logits, self.test_recurrent_hidden_states = self.model(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            actions = torch.argmax(logits, dim=1)

            self.prev_actions.copy_(actions)

        # Reset called externally, we're not done until then
        self.not_done_masks = torch.ones(1, 1, device=self.device, dtype=torch.bool)
        return actions[0].item(), semantic_vis


class EndToEndSemanticScout:
    """
    Environment setup on Apple M1 Mac:
    - fix default dependencies
        remove cudatoolkit, pytorch, and torchvision from fairo/conda.txt — we install them later
        remove habitat-sim from fairo/agents/locobot/conda.txt — we install it later
        comment out detectron2 in agents/locobot/requirements.txt — we install it later
    - create environment
        conda env remove -n droidlet -y
        mamba create -n droidlet python=3.8 --file conda.txt --file agents/locobot/conda.txt -c pytorch -c aihabitat -c conda-forge -y
        conda activate droidlet
    - install PyTorch
        pip uninstall scikit-image numpy scipy; pip install scikit-image numpy scipy — remove broken packages on Apple M1
        pip install torch torchvision — install PyTorch with functional numpy
    - install habitat-sim=0.2.0
        mamba install https://anaconda.org/aihabitat/habitat-sim/0.2.0/download/osx-64/habitat-sim-0.2.0-py3.8_osx_bfafd7934df465d79d807e4698659e2c20daf57d.tar.bz2
    - install habitat-lab==0.2.0 (version compatible with habitat-sim==0.2.0):
        git clone git@github.com:facebookresearch/habitat-lab.git
        pushd habitat-lab; git checkout tags/v0.2.0; pip install -r requirements.txt; python setup.py develop --all; popd
    - install the rest
        python setup.py develop
        pip install -r agents/locobot/requirements.txt
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

    If using MMDetection:
    - need to install PyTorch with official instructions:
        pip3 install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    - then install MMDetection
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.2/index.html
        git clone https://github.com/open-mmlab/mmdetection.git
        pushd mmdetection; pip install -r requirements/build.txt; pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"; pip install -v -e .; popd
    - then proceed with other instructions (can use pip install of conda for most installs if conda is too slow)
        conda install opencv -c pytorch -c conda-forge
        pip install -r agents/locobot/conda.txt
        conda install https://anaconda.org/aihabitat/habitat-sim/0.2.0/download/linux-64/habitat-sim-0.2.0-py3.8_headless_linux_bfafd7934df465d79d807e4698659e2c20daf57d.tar.bz2
        pushd habitat-lab; git checkout tags/v0.2.0; pip install -r requirements.txt; pip install -e .; popd
        python setup.py develop
        pip install -r agents/locobot/requirements.txt
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    """

    def __init__(self, mover, object_goal: str, episode_id: str, max_steps=400,
                 policy="original_camera_settings_and_mp3d_detector"):
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"

        self.path = f"trajectories/{episode_id}/end_to_end"
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path)
        os.makedirs(f"{self.path}/trajectory")

        self.max_steps = max_steps
        self.object_goal = object_goal
        self.object_goal_cat = coco_id_to_goal_id[coco_categories[object_goal]]

        self.in_habitat = isinstance(mover, LoCoBotMover)

        self.actions = {
            HabitatSimActions.MOVE_FORWARD: "forward",
            HabitatSimActions.TURN_RIGHT: "right",
            HabitatSimActions.TURN_LEFT: "left",
            HabitatSimActions.LOOK_UP: "up",
            HabitatSimActions.LOOK_DOWN: "down",
            HabitatSimActions.STOP: "stop",
        }

        this_dir = os.path.dirname(os.path.abspath(__file__))
        agent_config_file = this_dir + "/configs/rl_objectnav_sem_seg_hm3d.yaml"

        assert policy in [
            "original_camera_settings_and_mp3d_detector_il", 
            "robot_camera_settings_and_coco_detector_il",
            "robot_camera_settings_without_noise_and_coco_detector_il",
            "original_camera_settings_and_mp3d_detector_rl", 
            "robot_camera_settings_and_coco_detector_rl",
            "robot_camera_settings_without_noise_and_coco_detector_rl",
        ]
        if policy == "original_camera_settings_and_mp3d_detector_il":
            # Original camera settings + MP3D segmentation with IL training only
            challenge_config_file = this_dir + "/configs/original_camera_settings.yaml"
            model_path = this_dir + "/ckpt/original_camera_settings_and_mp3d_detector_il_ckpt28.pth"
        elif policy == "original_camera_settings_and_mp3d_detector_rl":
            # Original camera settings + MP3D segmentation with RL fine-tuning
            challenge_config_file = this_dir + "/configs/original_camera_settings.yaml"
            model_path = this_dir + "/ckpt/original_camera_settings_and_mp3d_detector_rl_ckpt32.pth"
        elif policy == "robot_camera_settings_and_coco_detector_il":
            # Robot camera settings + COCO segmentation with IL training only
            challenge_config_file = this_dir + "/configs/robot_camera_settings.yaml"
            model_path = this_dir + "/ckpt/robot_camera_settings_and_coco_detector_il_ckpt16.pth"
        elif policy == "robot_camera_settings_and_coco_detector_rl":
            # Robot camera settings + COCO segmentation with RL fine-tuning
            challenge_config_file = this_dir + "/configs/robot_camera_settings.yaml"
            model_path = this_dir + "/ckpt/robot_camera_settings_and_coco_detector_rl_ckpt14.pth"
        elif policy == "robot_camera_settings_without_noise_and_coco_detector_il":
            # Robot camera settings without noise + COCO segmentation with IL training only
            challenge_config_file = this_dir + "/configs/robot_camera_settings.yaml"
            model_path = this_dir + "/ckpt/robot_camera_settings_without_noise_and_coco_detector_il_ckpt24.pth"
        elif policy == "robot_camera_settings_without_noise_and_coco_detector_rl":
            # Robot camera settings without noise + COCO segmentation with RL fine-tuning
            challenge_config_file = this_dir + "/configs/robot_camera_settings.yaml"
            model_path = this_dir + "/ckpt/robot_camera_settings_without_noise_and_coco_detector_rl_ckpt21.pth"

        config = get_config(agent_config_file, ["BASE_TASK_CONFIG_PATH", challenge_config_file])
        config.defrost()
        config.MODEL_PATH = model_path
        config.MODEL.SEMANTIC_ENCODER.rednet_ckpt = (
            this_dir + "/" + config.MODEL.SEMANTIC_ENCODER.rednet_ckpt
        )
        config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
            this_dir + "/" + config.MODEL.DEPTH_ENCODER.ddppo_checkpoint
        )
        if torch.cuda.is_available():
            config.TORCH_GPU_ID = 0
        else:
            config.TORCH_GPU_ID = -1
        config.POLICY = policy
        config.freeze()
        self.config = config

        self.agent = RLSegFTAgent(config)

        self.step_count = 0
        self.finished = False
        self.agent.reset()
        self.start_time = time.time()

        self.semantic_frame = None
        self.pose = None
        self.all_poses = []
        self.action = None
        self.all_actions = []
        self.collision = None
        self.num_collisions = 0

    def step(self, mover):
        self.step_count += 1
        print("Step", self.step_count)

        if self.in_habitat:
            pose = mover.bot.get_base_state()
        else:
            pose = mover.bot.get_base_state().value

        gps = np.array([pose[0], -pose[1]], dtype=np.float32)
        compass = np.array(pose[2], dtype=np.float32)

        def preprocess_depth(depth):
            print("pre-processing: depth.min(), depth.max()", (depth.min(), depth.max()))
            min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
            max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
            clipped_depth = np.where(
                depth > 0,
                np.clip(depth, min_depth, max_depth),
                depth
            )
            rescaled_depth = np.where(
                clipped_depth > 0,
                (clipped_depth - min_depth) / (max_depth - min_depth),
                clipped_depth
            )
            rescaled_depth = np.expand_dims(rescaled_depth, -1).astype(np.float32)
            print(f"post-processing: depth.min(), depth.max()", (clipped_depth.min(), clipped_depth.max()))
            return rescaled_depth, clipped_depth

        if self.in_habitat:
            # Habitat
            rgb_depth = mover.get_rgb_depth()
            rgb = rgb_depth.rgb
            depth = rgb_depth.depth
        else:
            # Robot
            rgb, depth = mover.get_rgb_depth_optimized_for_habitat_transfer()

        def reshape_640x480_to_480x640(rgb, depth):
            print("pre-processing: frame shape", rgb.shape)
            # (640, 480) -> (360, 480)
            rgb = rgb[280:, :]
            depth = depth[280:, :]
            # (360, 480) -> (480, 640)
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
            print("post-processing: frame shape", rgb.shape)
            return rgb, depth

        def reshape_512x512_to_640x480(rgb, depth):
            # (512, 512) -> (512, 384)
            rgb = rgb[:, 64:448]
            depth = depth[280:, 64:448]
            # (512, 384) -> (640, 480)
            rgb = cv2.resize(rgb, (480, 640), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (480, 640), interpolation=cv2.INTER_LINEAR)
            return rgb, depth

        if ("original_camera_settings" in self.config.POLICY and 
                (rgb.shape[0] == 640 and rgb.shape[1] == 480)):
            rgb, depth = reshape_640x480_to_480x640(rgb, depth)

        depth, clipped_depth = preprocess_depth(depth)

        # obs = {
        #     "objectgoal": 0,
        #     "gps": np.zeros(2, dtype=np.float32),
        #     "compass": np.zeros(1, dtype=np.float32),
        #     "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "depth": np.zeros((480, 640, 1), dtype=np.float32),
        # }
        obs = {
            "objectgoal": self.object_goal_cat,
            "gps": gps,
            "compass": compass,
            "rgb": rgb,
            "depth": depth,
        }

        t0 = time.time()
        action, semantic_frame = self.agent.act(obs)
        t1 = time.time()

        forward_dist = 0.25
        turn_angle = 30
        tilt_angle = 30

        # Low-level actions
        print(f"Action: {self.actions.get(action)}")
        if self.in_habitat:
            # Habitat
            if action == HabitatSimActions.MOVE_FORWARD:
                status = mover.bot.go_to_relative((forward_dist, 0, 0), wait=True)
            elif action == HabitatSimActions.TURN_RIGHT:
                status = mover.bot.go_to_relative((0, 0, np.radians(-turn_angle)), wait=True)
            elif action == HabitatSimActions.TURN_LEFT:
                status = mover.bot.go_to_relative((0, 0, np.radians(turn_angle)), wait=True)
            elif action == HabitatSimActions.STOP:
                self.finished = True
                status = "SUCCEEDED"
            else:
                print("Action not implemented yet!")
                status = "SUCCEEDED"
        else:
            # Robot
            if action in [
                HabitatSimActions.MOVE_FORWARD,
                HabitatSimActions.TURN_RIGHT,
                HabitatSimActions.TURN_LEFT,
                HabitatSimActions.LOOK_UP,
                HabitatSimActions.LOOK_DOWN
            ]:
                result = safe_call(
                    mover.nav.execute_low_level_command, 
                    action, 
                    forward_dist, 
                    np.radians(turn_angle),
                    np.radians(tilt_angle)
                )
                status = result.value
            elif action == HabitatSimActions.STOP:
                self.finished = True
                status = "SUCCEEDED"
            else:
                print("Action not implemented yet!")
                status = "SUCCEEDED"

        print(f"Time {t1 - t0:.2f}")
        print()

        # TODO Can we use localization to enforce deterministic actions
        #  with the same effect as in simulation (exactly 25cm forward and
        #  exactly 30 degree turns)?

        # Visualization
        collision = status != "SUCCEEDED"
        self.snapshot(rgb, clipped_depth, semantic_frame,
                      pose, self.actions.get(action), collision)

        if self.step_count > self.max_steps - 1:
            self.finished = True

        if self.finished:
            if self.in_habitat:
                pose = mover.bot.get_base_state()
            else:
                pose = mover.bot.get_base_state().value
            self.record_aggregate_metrics(pose)

    def snapshot(self,
                 rgb_frame, depth_frame, semantic_frame,
                 start_pose, action, collision):
        self.semantic_frame = semantic_frame
        self.all_poses.append(start_pose)
        self.all_actions.append(action)
        if collision:
            self.num_collisions += 1

        snapshot_path = f"{self.path}/trajectory/step{self.step_count}"
        os.makedirs(snapshot_path)
        os.makedirs(f"{snapshot_path}/frames")

        # Frames
        cv2.imwrite(f"{snapshot_path}/frames/rgb.png", rgb_frame[:, :, ::-1])
        cv2.imwrite(f"{snapshot_path}/frames/depth.png",
                    ((depth_frame / depth_frame.max()) * 255).astype(np.uint8))
        np.save(f"{snapshot_path}/frames/depth.npy", depth_frame)
        cv2.imwrite(f"{snapshot_path}/frames/semantic.png", semantic_frame)

        # Metrics
        json.dump(
            {
                "timestamp": time.time() - self.start_time,
                "start_pose": start_pose,
                "action": action,
                "collision": collision,
            },
            open(f"{snapshot_path}/logs.json", "w")
        )

    def record_aggregate_metrics(self, last_pose):
        self.all_poses.append(last_pose)
        path_length = sum([
            np.linalg.norm(np.abs(np.array(end[:2]) - np.array(start[:2])))
            for start, end in zip(self.all_poses[:-1], self.all_poses[1:])
        ])
        json.dump(
            {
                "time": time.time() - self.start_time,
                "path_length": path_length,
                "num_steps": len(self.all_actions),
                "num_collisions": self.num_collisions,
                "poses": self.all_poses,
                "actions": self.all_actions,
            },
            open(f"{self.path}/aggregate_logs.json", "w")
        )
