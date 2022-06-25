import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from gym.spaces import Dict, Box
from .models.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
)
from .models.rnn_state_encoder import RNNStateEncoder
from .models.common import CategoricalNet, CustomFixedCategorical
from .policy import Net

from habitat import Config, logger


task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    11,  # ('bed', 10, 6)
    14,  # ('plant', 13, 8)
    18,  # ('toilet', 17, 10),
    22,  # ('tv_monitor', 21, 13)
    10,  # ('sofa', 9, 5),
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}


SEMANTIC_EMBEDDING_SIZE = 4


class SemSegSeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        model_config: Config,
        num_actions,
        device,
        goal_sensor_uuid=None,
        additional_sensors=["gps", "compass"],
    ):
        super().__init__()
        self.model_config = model_config

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
            "ResnetRGBEncoder",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
            )

        sem_seg_output_size = 0
        self.semantic_predictor = None
        if model_config.USE_SEMANTICS:
            # self.semantic_predictor = load_rednet(
            #     device,
            #     ckpt=model_config.SEMANTIC_ENCODER.rednet_ckpt,
            #     resize=True # since we train on half-vision
            # )
            # # Disable gradients
            # for param in self.semantic_predictor.parameters():
            #     param.requires_grad_(False)
            # self.semantic_predictor.eval()

            sem_embedding_size = model_config.SEMANTIC_ENCODER.embedding_size
            self.is_thda = model_config.SEMANTIC_ENCODER.is_thda

            rgb_shape = observation_space.spaces["rgb"].shape
            spaces = {
                "semantic": Box(
                    low=0,
                    high=255,
                    shape=(rgb_shape[0], rgb_shape[1], sem_embedding_size),
                    dtype=np.uint8,
                ),
            }
            sem_obs_space = Dict(spaces)
            self.sem_seg_encoder = ResnetSemSeqEncoder(
                sem_obs_space,
                output_size=model_config.SEMANTIC_ENCODER.output_size,
                backbone=model_config.SEMANTIC_ENCODER.backbone,
                trainable=model_config.SEMANTIC_ENCODER.train_encoder,
                semantic_embedding_size=sem_embedding_size,
            )
            sem_seg_output_size = model_config.SEMANTIC_ENCODER.output_size
            logger.info("Setting up Sem Seg model")

        # Init the RNN state decoder
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
            + sem_seg_output_size
        )

        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors

        if "gps" in additional_sensors:
            input_gps_dim = observation_space.spaces["gps"].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if "compass" in observation_space.spaces:
            assert (
                observation_space.spaces["compass"].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if self.goal_sensor_uuid is not None and self.goal_sensor_uuid != "no_sensor":
            self._n_object_categories = (
                int(observation_space.spaces[self.goal_sensor_uuid].high[0]) + 1
            )

            if self.is_thda:
                self._n_object_categories += 7
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.embed_sge = model_config.embed_sge
        if self.embed_sge:
            self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=device)
            self.mapping_mpcat40_to_goal = np.zeros(
                max(
                    max(mapping_mpcat40_to_goal21.keys()) + 1,
                    50,
                ),
                dtype=np.int8,
            )

            for key, value in mapping_mpcat40_to_goal21.items():
                self.mapping_mpcat40_to_goal[key] = value
            self.mapping_mpcat40_to_goal = torch.tensor(
                self.mapping_mpcat40_to_goal, device=device
            )
            rnn_input_size += 1

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_semantic_observations(self, observations_batch):
        with torch.no_grad():
            semantic_observations = self.semantic_predictor(
                observations_batch["rgb"], observations_batch["depth"]
            )
            return semantic_observations

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].flatten(start_dim=1)
            idx = self.task_cat2mpcat40[observations["objectgoal"].long()]
            if self.is_thda:
                # logger.info("idx: {}".format(idx.shape))
                idx = self.mapping_mpcat40_to_goal[idx].long()
                # logger.info("idx: {}".format(idx.shape))
            idx = idx.to(obj_semantic.device)

            goal_visible_pixels = (obj_semantic == idx.squeeze(0)).sum(
                dim=1
            )  # Sum over all since we're not batched
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))

            return goal_visible_area.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

        if len(rgb_obs.size()) == 5:
            observations["rgb"] = rgb_obs.contiguous().view(
                -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
            )

        if len(depth_obs.size()) == 5:
            observations["depth"] = depth_obs.contiguous().view(
                -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
            )

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        x = [depth_embedding, rgb_embedding]

        if self.model_config.USE_SEMANTICS:
            if self.model_config.embed_sge:
                sge_embedding = self._extract_sge(observations)
                x.append(sge_embedding)
            sem_seg_embedding = self.sem_seg_encoder(observations)
            x.append(sem_seg_embedding)

        if "gps" in self.additional_sensors:
            obs_gps = observations["gps"]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if "compass" in self.additional_sensors:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if self.goal_sensor_uuid is not None and self.goal_sensor_uuid != "no_sensor":
            object_goal = observations["objectgoal"].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))

            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class SemSegSeqHM3DModel(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
        device,
        goal_sensor_uuid=None,
        additional_sensors=["gps", "compass"],
    ):
        super().__init__()
        self.net = SemSegSeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
            device=device,
            goal_sensor_uuid=goal_sensor_uuid,
            additional_sensors=additional_sensors,
        )
        self.action_distribution = CategoricalNet(self.net.output_size, action_space.n)
        self.train()

    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution.logits, rnn_hidden_states
