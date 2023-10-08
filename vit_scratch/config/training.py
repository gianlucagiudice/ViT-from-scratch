from dataclasses import dataclass


@dataclass
class TrainingConfig:
    experiment_name: str
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping: int = 15
    seed: int = 42
    wandb_logger: bool = False
    batch_size: int = 256
