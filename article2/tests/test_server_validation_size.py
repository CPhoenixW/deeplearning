from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.config import FedConfig, apply_fed_config_overrides
from src.tasks import _split_train_test_loaders, _validation_indices


class _ToyDataset(Dataset):
    def __init__(self, labels: torch.Tensor) -> None:
        self.targets = labels.clone()

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        return torch.zeros(1), int(self.targets[index].item())


def _balanced_labels(num_classes: int = 10, samples_per_class: int = 20) -> torch.Tensor:
    return torch.arange(num_classes, dtype=torch.long).repeat_interleave(samples_per_class)


def test_server_validation_default_is_50() -> None:
    assert FedConfig().server_validation_size == 50


def test_validation_indices_select_exact_balanced_50_samples() -> None:
    labels = _balanced_labels()
    indices = _validation_indices(
        labels,
        num_classes=10,
        seed=42,
        size=50,
    )

    assert len(indices) == 50
    counts = torch.bincount(labels[indices], minlength=10)
    assert counts.tolist() == [5] * 10


def test_validation_size_is_reproducible_and_sweepable() -> None:
    labels = _balanced_labels()

    first = _validation_indices(labels, num_classes=10, seed=7, size=23)
    second = _validation_indices(labels, num_classes=10, seed=7, size=23)
    larger = _validation_indices(labels, num_classes=10, seed=7, size=100)

    assert first == second
    assert len(first) == 23
    assert len(larger) == 100

    counts = torch.bincount(labels[first], minlength=10)
    assert int(counts.max() - counts.min()) <= 1


def test_validation_samples_are_withheld_from_clients() -> None:
    labels = _balanced_labels()
    train_dataset = _ToyDataset(labels)
    validation_dataset = _ToyDataset(labels)
    test_dataset = _ToyDataset(labels[:50])
    config = FedConfig(
        num_clients=5,
        num_benign=5,
        server_validation_size=50,
        batch_size=8,
        dirichlet_alpha=None,
        device="cpu",
    )

    client_loaders, validation_loader, _test_loader = _split_train_test_loaders(
        config,
        train_dataset,
        validation_dataset,
        test_dataset,
        num_classes=10,
    )

    assert len(validation_loader.dataset) == 50
    assert sum(len(loader.dataset) for loader in client_loaders) == len(labels) - 50

    validation_indices = set(validation_loader.dataset.indices)
    client_indices = {
        index
        for loader in client_loaders
        for index in loader.dataset.indices
    }
    assert validation_indices.isdisjoint(client_indices)


def test_validation_size_can_be_overridden_from_experiment_config() -> None:
    config = apply_fed_config_overrides(
        FedConfig(),
        {"server_validation_size": 100},
        source="test",
    )
    assert config.server_validation_size == 100


def test_validation_size_rejects_invalid_values() -> None:
    labels = _balanced_labels()

    for invalid in (0, -1, labels.numel() + 1):
        try:
            _validation_indices(labels, num_classes=10, seed=42, size=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected server validation size {invalid} to fail")
