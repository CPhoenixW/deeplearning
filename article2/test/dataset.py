from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

try:
    from .config import FedConfig
    from .tasks import Cifar10Task
except ImportError:
    from config import FedConfig
    from tasks import Cifar10Task


def build_cifar10_dataloaders(config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
    """Backward-compatible entry point; prefer ``get_task(config).build_dataloaders``."""

    return Cifar10Task().build_dataloaders(config)
