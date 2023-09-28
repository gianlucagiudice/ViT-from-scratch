from vit_scratch.models import ViTCustom
from vit_scratch.dataset import FashionMNISTDataModule
import pytorch_lightning as pl
import fire

from lightning.pytorch.loggers import MLFlowLogger


def main(
        data_dir: str = "./",
        batch_size: int = 256,
        patch_size: int = 7,
        latent_dim: int = 128,
        n_layers: int = 12,
        n_heads: int = 8,
        max_epochs: int = 100,
        early_stopping_patience: int = 15,
        accelerator: str = 'auto'
):
    fashion_mnist = FashionMNISTDataModule(data_dir, batch_size)
    fashion_mnist.prepare_data()
    fashion_mnist.setup('fit')

    # Get data loaders
    train_loader = fashion_mnist.train_dataloader()
    val_loader = fashion_mnist.val_dataloader()
    test_loader = fashion_mnist.test_dataloader()

    model = ViTCustom(
        input_shape=fashion_mnist.input_shape,
        patch_size=patch_size,
        latent_dim=latent_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_classes=fashion_mnist.n_classes,
    )

    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        verbose=True,
        mode='min'
    )
    # Tensorboard logger
    mlf_logger = MLFlowLogger(
        experiment_name='vit_scratch',
        tracking_uri='file:./mlruns',
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[early_stopping],
        log_every_n_steps=10,
        logger=mlf_logger,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    fire.Fire(main)
