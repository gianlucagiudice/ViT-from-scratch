import enum
from abc import ABC
from dataclasses import dataclass
from typing import Tuple


class ModelType(enum.Enum):
    resnet_baseline: str = 'resnet_baseline'
    vit_custom: str = 'vit_custom'


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
    dropout: float


@dataclass
class ResnetBaselineConfig(ModelConfig):
    num_channels: int
    resnet_size: int
    pretrained: bool
