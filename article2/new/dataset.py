from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import FedConfig


def build_cifar10_dataloaders(config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
    """Create IID client dataloaders and a global test loader."""

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset: Dataset = datasets.CIFAR10(root=config.data_root, train=True, download=True, transform=transform_train)
    test_dataset: Dataset = datasets.CIFAR10(root=config.data_root, train=False, download=True, transform=transform_test)

    num_clients = config.num_clients
    num_samples = len(train_dataset)
    indices = torch.randperm(num_samples).tolist()
    split_size = num_samples // num_clients

    client_loaders: List[DataLoader] = []
    for i in range(num_clients):
        start = i * split_size
        end = num_samples if i == num_clients - 1 else (i + 1) * split_size
        subset = Subset(train_dataset, indices[start:end])
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return client_loaders, test_loader

