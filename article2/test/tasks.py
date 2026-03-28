from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from .config import FedConfig
    from .models import resnet18_cifar10, resnet18_fashion_mnist
except ImportError:
    from config import FedConfig
    from models import resnet18_cifar10, resnet18_fashion_mnist


class FederatedTask(ABC):
    """One dataset + one backbone; plug in via TASK_REGISTRY."""

    name: str
    num_classes: int

    @abstractmethod
    def data_subdir(self, config: FedConfig) -> str:
        """Subfolder under config.data_root for this dataset."""

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Global model architecture for this task."""

    @abstractmethod
    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        """IID split: one train loader per client + one test loader."""


class Cifar10Task(FederatedTask):
    name = "cifar10"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "cifar10")

    def build_model(self) -> torch.nn.Module:
        return resnet18_cifar10(num_classes=self.num_classes)

    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor()])

        root = self.data_subdir(config)
        train_dataset: Dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        test_dataset: Dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(config, train_dataset, test_dataset)


class FashionMnistTask(FederatedTask):
    name = "fashion_mnist"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "fashion_mnist")

    def build_model(self) -> torch.nn.Module:
        return resnet18_fashion_mnist(num_classes=self.num_classes)

    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        # Resize to 32×32 to match ResNet18 small-image stem (same spatial size as CIFAR-10).
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

        root = self.data_subdir(config)
        train_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform_train
        )
        test_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(config, train_dataset, test_dataset)


def _split_train_test_loaders(
    config: FedConfig, train_dataset: Dataset, test_dataset: Dataset
) -> Tuple[List[DataLoader], DataLoader]:
    num_clients = config.num_clients
    num_samples = len(train_dataset)
    indices = torch.randperm(num_samples).tolist()
    split_size = num_samples // num_clients

    client_loaders: List[DataLoader] = []
    for i in range(num_clients):
        start = i * split_size
        end = num_samples if i == num_clients - 1 else (i + 1) * split_size
        subset = Subset(train_dataset, indices[start:end])
        loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )
    return client_loaders, test_loader


TASK_REGISTRY: Dict[str, Type[FederatedTask]] = {
    "cifar10": Cifar10Task,
    "fashion_mnist": FashionMnistTask,
}


def get_task(config: FedConfig) -> FederatedTask:
    key = config.task_name.lower().strip()
    cls = TASK_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown task_name: {config.task_name}. "
            f"Available: {sorted(TASK_REGISTRY.keys())}"
        )
    return cls()
