import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from vit_scratch.dataset import FashionMNISTDataModule
from vit_scratch.models import (
    ViTNetwork, ResnetBaselineNetwork,
    ViTConfig, ResnetBaselineConfig,
    ModelWrapper
)


def run_experiment(
        model: ModelWrapper,
        experiment_name: str,
        batch_size: int = 256,
        early_stopping_patience: int = 15,
        max_epochs: int = 100,
        data_dir: str = "./",
        accelerator: str = 'auto',
        seed: int = 42,
        wandb_logger: bool = False,
):
    fashion_mnist = FashionMNISTDataModule(data_dir, batch_size, seed=seed)
    fashion_mnist.prepare_data()
    fashion_mnist.setup('fit')

    # Get data loaders
    train_loader = fashion_mnist.train_dataloader()
    val_loader = fashion_mnist.val_dataloader()
    test_loader = fashion_mnist.test_dataloader()

    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        verbose=True,
        mode='min'
    )
    # Tensorboard logger
    if wandb_logger:
        logger = WandbLogger(
            project='ViT-Scratch',
            name=experiment_name,
        )
        # Log hyperparameters
        logger.experiment.config['model_type'] = model.model.__class__.__name__
        logger.experiment.config['batch_size'] = batch_size
    else:
        logger = TensorBoardLogger(
            save_dir='tb_logs',
            name=experiment_name
        )

    # Train model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[early_stopping],
        log_every_n_steps=10,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)

    # Test model
    trainer.test(model, test_loader)


def train_test_helper(
        # Model
        model: ModelWrapper,
        # Experiment config
        experiment_name: str,
        seed: int,
        max_epochs: int,
        wandb_logger: bool = False,
):
    run_experiment(
        model, experiment_name,
        seed=seed, max_epochs=max_epochs, wandb_logger=wandb_logger
    )


def train_test_resnet(
        # Resnet config
        resnet_size: int = 18,
        pretrained: bool = False,
        # Training config
        learning_rate: float = 1e-3,
        # Data config
        num_channels: int = 1,
        n_classes: int = 10,
        # Experiment config
        experiment_name: str = 'resnet_baseline',
        seed: int = 42,
        max_epochs: int = 100,
        wandb_logger: bool = False,
):
    model_config = ResnetBaselineConfig(
        num_channels=num_channels,
        resnet_size=resnet_size,
        pretrained=pretrained,
        n_classes=n_classes,
    )
    resnet_baseline = ResnetBaselineNetwork(model_config)
    model = ModelWrapper(resnet_baseline, learning_rate, model_config=model_config)
    # Run experiment
    train_test_helper(
        model, experiment_name,
        seed=seed, max_epochs=max_epochs, wandb_logger=wandb_logger)


def train_test_vit(
        # ViT config
        patch_size: int = 7,
        latent_dim: int = 48,
        n_layers: int = 12,
        n_heads: int = 8,
        # Training config
        learning_rate: float = 1e-3,
        # Data config
        input_channels: int = 1,
        n_classes: int = 10,
        image_size: int = 28,
        # Experiment config
        experiment_name: str = 'vit_custom',
        seed: int = 42,
        max_epochs: int = 100,
        wandb_logger: bool = False,
):
    model_config = ViTConfig(
        input_shape=(input_channels, image_size, image_size),
        patch_size=patch_size,
        latent_dim=latent_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_classes=n_classes,
    )
    vit_network = ViTNetwork(model_config)
    model = ModelWrapper(vit_network, learning_rate, model_config=model_config)
    # Run experiment
    train_test_helper(
        model, experiment_name,
        seed=seed, max_epochs=max_epochs, wandb_logger=wandb_logger)
