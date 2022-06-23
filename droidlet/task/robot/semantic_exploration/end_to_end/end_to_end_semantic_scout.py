import numpy as np
import torch
import os
import time

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
from droidlet.perception.robot.semantic_mapper.constants import coco_categories


class RLSegFTAgent(Agent):
    def __init__(self, config: Config):
        if not config.MODEL_PATH:
            raise Exception("Model checkpoint wasn't provided, quitting.")
        if config.TORCH_GPU_ID >= 0:
            self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        else:
            self.device = torch.device("cpu")

        ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)["state_dict"]
        ckpt_dict = {k.replace("actor_critic.", ""): v for k, v in ckpt_dict.items()}
        ckpt_dict = {k.replace("module.", ""): v for k, v in ckpt_dict.items()}

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

        self.model.load_state_dict(ckpt_dict, strict=True)
        self.model.eval()

        self.semantic_predictor = None

        if self.model_cfg.USE_SEMANTICS:
            logger.info("setting up sem seg predictor")
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=self.model_cfg.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True,  # Since we train on half-vision
                num_classes=self.model_cfg.SEMANTIC_ENCODER.num_classes,
            )
            self.semantic_predictor.eval()

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

    @torch.no_grad()
    def act(self, observations):

        batch = batch_obs([observations], device=self.device)

        with torch.no_grad():
            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1
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
        return actions[0].item()


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
    """

    def __init__(self, mover, object_goal: str, max_steps=400):
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"

        self.max_steps = max_steps
        self.object_goal = object_goal
        coco_id_to_habitat_id = {
            0: 0,  # chair
            1: 5,  # couch
            2: 2,  # potted plant
            3: 1,  # bed
            4: 3,  # toilet
            5: 4,  # tv
        }
        self.object_goal_cat = coco_id_to_habitat_id[coco_categories[object_goal]]

        this_dir = os.path.dirname(os.path.abspath(__file__))
        challenge_config_file = this_dir + "/configs/challenge_objectnav2022.local.rgbd.yaml"
        agent_config_file = this_dir + "/configs/rl_objectnav_sem_seg_hm3d.yaml"
        model_path = this_dir + "/ckpt/model.pth"
        config = get_config(agent_config_file, ["BASE_TASK_CONFIG_PATH", challenge_config_file])
        config.defrost()
        config.MODEL_PATH = model_path
        config.MODEL.SEMANTIC_ENCODER.rednet_ckpt = (
            this_dir + "/" + config.MODEL.SEMANTIC_ENCODER.rednet_ckpt
        )
        config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
            this_dir + "/" + config.MODEL.DEPTH_ENCODER.ddppo_checkpoint
        )
        config.TORCH_GPU_ID = 1
        config.freeze()

        self.agent = RLSegFTAgent(config)

        self.step_count = 0
        self.finished = False
        self.agent.reset()

    def step(self, mover):
        self.step_count += 1
        print("Step", self.step_count)

        pose = mover.bot.get_base_state()
        gps = np.array(pose[:2], dtype=np.float32)
        compass = np.array(pose[2], dtype=np.float32)

        def preprocess_depth(depth, min_depth=0.5, max_depth=5.0):
            depth = np.clip(depth, min_depth, max_depth)
            depth = (depth - min_depth) / (max_depth - min_depth)
            depth = np.expand_dims(depth, -1)
            return depth

        rgb_depth = mover.get_rgb_depth()
        rgb = rgb_depth.rgb
        depth = rgb_depth.depth

        print("objectgoal", self.object_goal_cat)
        print("gps", gps)
        print("compass", compass)

        print("before: depth.min(), depth.max()", (depth.min(), depth.max()))
        depth = preprocess_depth(depth)
        print("after: depth.min(), depth.max()", (depth.min(), depth.max()))

        # TODO Set Habitat camera parameters to Habitat Challenge to gain confidence
        #  in policy:
        #  done - (480, 640) instead of (512, 512)
        #  hacky done - [0.5, 5.0] depth range instead of [0.0, 10.0]
        #  try without - 79 HFOV instead of 90
        #  done- 0.88 camera height instead of 0.6 (is it 0.6?)

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
        action = self.agent.act(obs)
        t1 = time.time()

        # TODO Compute relative (x, y, yaw) from action
        forward_dist = 0.25
        turn_angle = 30

        if action == HabitatSimActions.MOVE_FORWARD:
            print("Action: forward")
            x = forward_dist
            y, yaw = 0, 0
        elif action == HabitatSimActions.TURN_RIGHT:
            print("Action: right")
            x, y = 0, 0
            yaw = np.radians(-turn_angle)
        elif action == HabitatSimActions.TURN_LEFT:
            print("Action: left")
            x, y = 0, 0
            yaw = np.radians(turn_angle)

        print(f"Time {t1 - t0:.2f}")
        print()

        mover.bot.go_to_relative((x, y, yaw), wait=True)

        # TODO Can we use localization to enforce deterministic actions
        #  with the same effect as in simulation (exactly 25cm forward and
        #  exactly 30 degree turns)?

        if self.step_count > self.max_steps:
            self.finished = True
