from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseModel(pl.LightningModule, ABC):

    @abstractmethod
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            learning_rate: float = 0.001,
            *args, **kwargs
    ):
        super().__init__()
        # Set the model
        self.model = model
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, step):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        # Get probability predictions
        outputs_probs = torch.softmax(outputs, dim=1)
        # Get accuracy
        acc = (outputs_probs.argmax(dim=1) == targets).float().mean()
        self.log(f'{step}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'val')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

