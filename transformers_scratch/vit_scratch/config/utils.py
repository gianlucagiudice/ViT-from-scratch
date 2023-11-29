from typing import Tuple


def parse_config(config_path: str) -> Tuple[dict, dict]:
    import yaml
    config = yaml.safe_load(open(config_path, 'r'))

    if (model_params := config.get('model_config', None)) is None:
        raise ValueError("model_config key not found in config file")
    if (training_params := config.get('training_config', None)) is None:
        raise ValueError("model_config key not found in config file")

    return model_params, training_params
