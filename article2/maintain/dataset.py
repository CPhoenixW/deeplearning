# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - 数据集与 IID 划分
CIFAR-10，按客户端数量做 IID 划分。
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from typing import List, Tuple


# CIFAR-10 常用归一化
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(train: bool) -> Compose:
    """训练/测试的 transform。"""
    t = [ToTensor(), Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)]
    return Compose(t)


def get_cifar10_full_dataset(root: str = "./data", train: bool = True) -> CIFAR10:
    """获取完整 CIFAR-10 数据集。"""
    return CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=get_cifar10_transforms(train),
    )


def iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    IID 划分：将样本索引随机打乱后均分给 num_clients 个客户端。

    Args:
        dataset: 完整数据集（按 len(dataset) 取索引）。
        num_clients: 客户端数量 K。
        seed: 随机种子。

    Returns:
        indices_per_client: 长度为 K 的列表，每项为该客户端拥有的样本索引列表。
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    indices_per_client: List[List[int]] = [[] for _ in range(num_clients)]
    for i, idx in enumerate(perm):
        indices_per_client[i % num_clients].append(int(idx))
    return indices_per_client


def get_client_dataloaders(
    num_clients: int,
    batch_size: int = 32,
    root: str = "./data",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[List[DataLoader], DataLoader, CIFAR10]:
    """
    生成每个客户端的训练 DataLoader 与全局测试 DataLoader。

    Args:
        num_clients: 客户端数量 K。
        batch_size: 本地训练 batch size。
        root: CIFAR-10 数据根目录。
        seed: IID 划分随机种子。
        num_workers: DataLoader 的 num_workers。

    Returns:
        client_loaders: 长度为 K 的 DataLoader 列表。
        test_loader: 全局测试集 DataLoader。
        test_dataset: 测试集实例（用于评估）。
    """
    train_dataset = get_cifar10_full_dataset(root=root, train=True)
    test_dataset = get_cifar10_full_dataset(root=root, train=False)

    indices_per_client = iid_partition(train_dataset, num_clients, seed=seed)
    client_loaders: List[DataLoader] = []
    for indices in indices_per_client:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
    )
    return client_loaders, test_loader, test_dataset
