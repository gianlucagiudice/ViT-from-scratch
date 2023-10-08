import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from vit_scratch.config import ViTConfig, ResnetBaselineConfig, TrainingConfig
from vit_scratch.dataset import FashionMNISTDataModule
from vit_scratch.models import ViTNetwork, ResnetBaselineNetwork, ModelWrapper


def run_experiment(
        model: ModelWrapper,
        experiment_name: str,
        batch_size: int,
        early_stopping_patience: int,
        max_epochs: int,
        seed: int,
        wandb_logger: bool,
        data_dir: str = "./",
        accelerator: str = 'auto',
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
        wandb.init(project='ViT-Scratch', name=experiment_name)
        wandb.config.update({
            'model_type': model.model.__class__.__name__
        }, allow_val_change=True)
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

    # Close wandb
    if wandb_logger:
        wandb.finish()


def train_test_helper(
        # Model
        model: ModelWrapper,
        # Training config
        training_config: TrainingConfig,
):
    run_experiment(
        model,
        experiment_name=training_config.experiment_name,
        seed=training_config.seed,
        max_epochs=training_config.max_epochs,
        wandb_logger=training_config.wandb_logger,
        early_stopping_patience=training_config.early_stopping,
        batch_size=training_config.batch_size,
    )


def train_test_resnet(
        # Resnet config
        model_config: ResnetBaselineConfig,
        # Training config
        training_config: TrainingConfig,
):
    resnet_baseline = ResnetBaselineNetwork(model_config)
    model = ModelWrapper(
        resnet_baseline, learning_rate=training_config.learning_rate, model_config=model_config)
    # Run experiment
    train_test_helper(model, training_config)


def train_test_vit(
        # ViT config
        model_config: ViTConfig,
        # Training config
        training_config: TrainingConfig,
):
    vit_network = ViTNetwork(model_config)
    model = ModelWrapper(
        vit_network, learning_rate=training_config.learning_rate, model_config=model_config)
    # Run experiment
    train_test_helper(model, training_config)
