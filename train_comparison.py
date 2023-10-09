import fire

from vit_scratch.config import parse_config, TrainingConfig, ResnetBaselineConfig, ViTConfig
from vit_scratch.trainer import train_test_vit, train_test_resnet


def main(vit_config_path: str, resnet_config_path: str, n_seeds: int = 5):
    # Parse config
    vit_model_config, vit_training_config = parse_config(vit_config_path)
    resnet_model_config, resnet_training_config = parse_config(resnet_config_path)

    for seed in range(n_seeds):

        # Override config
        shared_training_config = {
            'seed': seed,
            'wandb_logger': True,
        }

        print(f'============ Running seed {seed} ============\n')

        # Run ViT experiment
        vit_training_config.update(shared_training_config)
        vit_training_config['experiment_name'] = f'vit_custom_seed-{seed}'
        model_config = ViTConfig(**vit_model_config)
        training_config = TrainingConfig(**vit_training_config)
        train_test_vit(model_config, training_config)

        # Run Resnet experiment
        resnet_training_config.update(shared_training_config)
        resnet_training_config['experiment_name'] = f'resnet_baseline_seed-{seed}'
        model_config = ResnetBaselineConfig(**resnet_model_config)
        training_config = TrainingConfig(**resnet_training_config)
        train_test_resnet(model_config, training_config)

        print(f'=========================================\n\n\n')


if __name__ == '__main__':
    fire.Fire(main)
