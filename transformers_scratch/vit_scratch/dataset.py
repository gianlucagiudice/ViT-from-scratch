import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import FashionMNIST


class FashionMNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = "./",
            batch_size: int = 32,
            resize_shape=None,
            num_workers: int = 8,
            seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = self._transform_factory(resize_shape)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Create split for train/val/test datasets
        self.fashion_mnist_full = None
        self.fashion_mnist_train = None
        self.fashion_mnist_val = None
        self.fashion_mnist_test = None
        self.seed = seed
        # Metadata
        self.n_classes = 10
        self.input_shape = (1, 28, 28)
        self.num_channels = self.input_shape[0]

    @staticmethod
    def _transform_factory(resize_shape=None):
        transforms_list = []
        if resize_shape is not None:
            transforms_list.append(transforms.Resize(resize_shape))

        transforms_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(0, 1),
            ]
        )

        return transforms.Compose(transforms_list)

    def prepare_data(self):
        # download
        try:
            FashionMNIST(self.data_dir, download=False)
        except RuntimeError:
            FashionMNIST(self.data_dir, download=True)

    def setup(
            self,
            stage: str,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1
    ):
        if stage == "fit" or stage is None:
            if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
                raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

            self.fashion_mnist_full = FashionMNIST(self.data_dir, transform=self.transform)

            # Get the number of images in the dataset (replace this with your own dataset)
            num_images = len(self.fashion_mnist_full)  # You need to define 'dataset' or your data source here

            # Calculate the number of images for each split based on ratios
            num_train = int(train_ratio * num_images)
            num_val = int(val_ratio * num_images)
            num_test = num_images - num_train - num_val

            # Randomly split the dataset into train, val, and test
            self.fashion_mnist_train, self.fashion_mnist_val, self.fashion_mnist_test = random_split(
                self.fashion_mnist_full, [num_train, num_val, num_test],
                generator=torch.Generator().manual_seed(self.seed)
            )

    def split_dataloader_helper(self, split):
        return DataLoader(split, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)

    def train_dataloader(self):
        return self.split_dataloader_helper(self.fashion_mnist_train)

    def val_dataloader(self):
        return self.split_dataloader_helper(self.fashion_mnist_val)

    def test_dataloader(self):
        return self.split_dataloader_helper(self.fashion_mnist_test)


if __name__ == '__main__':
    # Create the dataset
    fashion_mnist_data_module = FashionMNISTDataModule('../../')
    # Prepare the data
    fashion_mnist_data_module.prepare_data()
    fashion_mnist_data_module.setup(stage='fit')
    # Create the dataloaders
    train_loader = fashion_mnist_data_module.train_dataloader()
    val_loader = fashion_mnist_data_module.val_dataloader()
    test_loader = fashion_mnist_data_module.test_dataloader()
    # Print the size of each dataset
    print(f"Number of training examples: {len(train_loader.dataset)}")
    print(f"Number of validation examples: {len(val_loader.dataset)}")
    print(f"Number of test examples: {len(test_loader.dataset)}")
