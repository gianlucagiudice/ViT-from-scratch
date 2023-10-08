import fire
from vit_scratch.trainer import train_test_vit, train_test_resnet


def main():
    for seed in range(10):
        print(f'============ Running seed {seed} ============\n')
        # Run ViT experiment
        train_test_vit(
            experiment_name=f'vit_custom_seed-{seed}',
            seed=seed, wandb_logger=True,
        )
        # Run Resnet experiment
        train_test_resnet(
            experiment_name=f'resnet_baseline_seed-{seed}',
            seed=seed, wandb_logger=True,
        )

        print(f'=========================================\n\n\n')


if __name__ == '__main__':
    fire.Fire(main)
