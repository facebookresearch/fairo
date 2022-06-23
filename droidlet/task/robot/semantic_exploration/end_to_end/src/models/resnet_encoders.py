import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from gym import spaces
from .common import Flatten
from . import resnet
from typing import Dict


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["rgb"].shape[:2])
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            # spatial_size = observation_space.spaces["rgb"].shape[:2] // 2
            spatial_size = observation_space.spaces["rgb"].shape[:2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["depth"].shape[:2])
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            # spatial_size = observation_space.spaces["depth"].shape[:2] // 2
            spatial_size = observation_space.spaces["depth"].shape[:2]
        else:
            self._n_input_depth = 0
        
        if "semantic" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["semantic"].shape[:2])
            self._n_input_semantics = observation_space.spaces["semantic"].shape[2]
        else:
            self._n_input_semantics = 0
        
        if self._frame_size == (256, 256):
            spatial_size = (128, 128)
        elif self._frame_size == (240, 320):
            spatial_size = (120, 108)
        elif self._frame_size == (480, 640):
            spatial_size = (120, 108)

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb + self._n_input_semantics
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            # final_spatial = int(
            #     spatial_size * self.backbone.final_spatial_compress
            # )
            final_spatial = np.array([math.ceil(
                d * self.backbone.final_spatial_compress
            ) for d in spatial_size])
            after_compression_flat_size = 2048
            # num_compression_channels = int(
            #     round(after_compression_flat_size / (final_spatial ** 2))
            # )
            num_compression_channels = int(
                round(after_compression_flat_size / np.prod(final_spatial))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial[0],
                final_spatial[1],
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_semantics == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self._n_input_semantics > 0:
            semantic_observations = observations["semantic"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = semantic_observations.permute(0, 3, 1, 2)

            cnn_input.append(semantic_observations)

        x = torch.cat(cnn_input, dim=1)
        if self._frame_size == (256, 256):
            x = F.avg_pool2d(x, 2)
        elif self._frame_size == (240, 320):
            x = F.avg_pool2d(x, (2, 3), padding=(0, 1)) # 240 x 324 -> 120 x 108
        elif self._frame_size == (480, 640):
            x = F.avg_pool2d(x, (4, 5))

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)
        self.visual_encoder.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_depth = observations["depth"]
        if len(obs_depth.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )

        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, observation_space, output_size, device, spatial_output: bool = False
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.cnn.fc = nn.Sequential()
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        def resnet_forward(observation):
            # resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            # def hook(m, i, o):
            #     resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            # h = self.layer_extract.register_forward_hook(hook)
            # self.cnn(observation)
            # h.remove()
            # output: [BATCH x RESNET_DIM]
            resnet_output = self.cnn(observation)
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )  # [BATCH x OUTPUT_DIM]


class ResnetRGBEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"rgb": observation_space.spaces["rgb"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class ResnetSemSeqEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        backbone="resnet18",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
        semantic_embedding_size=4,
        use_pred_semantics=False,
        use_goal_seg=False,
    ):
        super().__init__()
        if not use_goal_seg:
            self.semantic_embedder = nn.Embedding(40 + 2, semantic_embedding_size)

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"semantic": observation_space.spaces["semantic"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output
        self.use_goal_seg = use_goal_seg

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_semantic = observations["semantic"]
        if len(obs_semantic.size()) == 5:
            observations["semantic"] = obs_semantic.contiguous().view(
                -1, obs_semantic.size(2), obs_semantic.size(3), obs_semantic.size(4)
            )

        if "semantic_features" in observations:
            x = observations["semantic_features"]
        else:
            # Embed input when using all object categories
            if not self.use_goal_seg:
                categories = observations["semantic"].long() + 1
                observations["semantic"] = self.semantic_embedder(categories)
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

