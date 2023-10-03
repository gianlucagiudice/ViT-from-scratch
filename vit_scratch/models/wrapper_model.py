from abc import abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


class ModelWrapper(pl.LightningModule):

    @abstractmethod
    def __init__(
            self,
            model: nn.Module,
            learning_rate: Optional[float],
            *args, **kwargs
    ):
        super().__init__()
        # Set the model
        self.model = model
        self.learning_rate = learning_rate
        # Save the model config
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, step, on_step: bool = True):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        # Get probability predictions
        outputs_probs = torch.softmax(outputs, dim=1)
        # Get accuracy
        acc = (outputs_probs.argmax(dim=1) == targets).float().mean()
        self.log(f'{step}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step}_accuracy', acc, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'test', on_step=False)
        return loss

    def configure_optimizers(self):
        if self.learning_rate is None:
            raise ValueError('Learning rate must be set')
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
