import fire

from transformers_scratch.vit_scratch.trainer import train_test_vit
from transformers_scratch.vit_scratch.config import (ViTConfig, ResnetBaselineConfig,
                                                     TrainingConfig, ModelType, parse_config)


def train_from_config(
        config_path: str = 'configs/vit/vit_custom.yaml',
        model_type: str = 'vit_custom',
        experiment_name: str = None,
):
    # Parse config
    model_config, training_config = parse_config(config_path)

    # Set model type
    try:
        model_type: ModelType = ModelType[model_type]
    except KeyError:
        raise ValueError(f"Model type {model_type} not supported."
                         f"Please choose from {list(ModelType)}")

    # Set experiment name
    if experiment_name is None:
        experiment_name = f"{model_type.value}-experiment"
    training_config['experiment_name'] = experiment_name

    # Run experiment
    training_config = TrainingConfig(**training_config)

    match model_type:
        case ModelType.vit_custom:
            model_config = ViTConfig(**model_config)
        case ModelType.resnet_baseline:
            model_config = ResnetBaselineConfig(**model_config)
        case _:
            raise ValueError(f"Model type {model_type} not supported")

    # Print model config
    print(f'Model config: {model_config}')

    # Start training
    train_test_vit(model_config=model_config, training_config=training_config)


if __name__ == '__main__':
    fire.Fire(train_from_config)
