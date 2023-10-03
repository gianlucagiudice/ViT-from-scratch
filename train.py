import fire

from vit_scratch.models import ModelType
from vit_scratch.trainer import train_test_vit, train_test_resnet
import yaml


def train_from_config(config_path: str, model_type: str = 'vit_custom'):
    config_params = yaml.safe_load(open(config_path, 'r'))

    try:
        model_type = ModelType[model_type]
    except KeyError:
        raise ValueError(f"Model type {model_type} not supported."
                         f"Please choose from {list(ModelType)}")

    match model_type:
        case ModelType.vit_custom:
            train_test_vit(**config_params)
        case ModelType.resnet_baseline:
            train_test_resnet(**config_params)
        case _:
            raise ValueError(f"Model type {model_type} not supported")


if __name__ == '__main__':
    fire.Fire(train_from_config)
