from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms


class CifarModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        dataset: str = "CIFAR10",  # Choose between "CIFAR10" or "CIFAR100"
        train_val_test_split: Tuple[int, int, int] = (45_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_labels: int = 100,  # The number of labels to keep per class
    ) -> None:
        """Initialize a `CifarModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param dataset: The dataset to use ("CIFAR10" or "CIFAR100"). Defaults to `"CIFAR10"`.
        :param train_val_test_split: The train, validation, and test split. Defaults to `(45_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param num_labels: The total number of labels to keep. Must be greater than 0.
        """
        super().__init__()

        if num_labels <= 0:
            raise ValueError("num_labels must be greater than 0.")

        self.save_hyperparameters(logger=False)

        # Data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

        self.batch_size_per_device = batch_size

        if self.hparams.dataset == "CIFAR10":
            self.dataset_class = CIFAR10
            self.num_classes = 10
        elif self.hparams.dataset == "CIFAR100":
            self.dataset_class = CIFAR100
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {self.hparams.dataset}")

        self.num_labels_per_class = self.hparams.num_labels // self.num_classes

    def prepare_data(self) -> None:
        """Download data if needed."""
        self.dataset_class(self.hparams.data_dir, train=True, download=True)
        self.dataset_class(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and split into train/val/test sets."""
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = self.dataset_class(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = self.dataset_class(self.hparams.data_dir, train=False, transform=self.transforms)

            # Filter to retain only num_labels_per_class labels and set others to -1
            if self.hparams.num_labels > 0:
                targets = torch.tensor(trainset.targets)
                mask = torch.zeros_like(targets, dtype=torch.bool)
                
                for c in range(self.num_classes):
                    class_indices = (targets == c).nonzero(as_tuple=True)[0]
                    selected_indices = class_indices[:self.num_labels_per_class]
                    mask[selected_indices] = True

                # Set the labels of non-selected samples to -1
                targets[~mask] = -1
                trainset.targets = targets.tolist()

            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the datamodule state."""
        pass


if __name__ == "__main__":
    _ = CifarModule(num_labels=500)
