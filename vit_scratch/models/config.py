from abc import ABC
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig(ABC):
    n_classes: int


@dataclass
class ViTConfig(ModelConfig):
    input_shape: Tuple[int, int, int]
    patch_size: int
    latent_dim: int
    n_layers: int
    n_heads: int


@dataclass
class ResnetBaselineConfig(ModelConfig):
    num_channels: int
    resnet_size: int
    pretrained: bool
