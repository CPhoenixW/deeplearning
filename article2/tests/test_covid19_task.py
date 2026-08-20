from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from src.config import FedConfig
from src.models import Covid19ResNet50
from src.tasks import (
    COVID19_CLASS_TO_INDEX,
    TASK_REGISTRY,
    _stratified_train_test_indices,
    scan_covid19_records,
)


SOURCE_DIRS = {
    "COVID": "COVID",
    "Lung_Opacity": "Lung_Opacity",
    "Normal": "Normal",
    "Viral_Pneumonia": "Viral Pneumonia",
}


def _write_fixture(root: Path, samples_per_class: int = 20) -> Path:
    dataset_root = root / "covid19" / "COVID-19_Radiography_Dataset"
    for class_name, source_dir in SOURCE_DIRS.items():
        image_dir = dataset_root / source_dir / "images"
        mask_dir = dataset_root / source_dir / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        value = 32 + 40 * COVID19_CLASS_TO_INDEX[class_name]
        for index in range(samples_per_class):
            image = Image.new("L", (16, 12), color=value)
            image.save(image_dir / f"{class_name}_{index:03d}.png")
        Image.new("L", (16, 12), color=255).save(mask_dir / "ignored_mask.png")
    return dataset_root


def test_covid19_model_uses_pretrained_resnet50_with_frozen_backbone() -> None:
    task = TASK_REGISTRY["covid19"]()
    model = task.build_model()
    assert isinstance(model, Covid19ResNet50)
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) > 20_000_000
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 8196


def test_covid19_fixed_mapping_and_scanner_ignore_masks(tmp_path: Path) -> None:
    dataset_root = _write_fixture(tmp_path, samples_per_class=3)
    records = scan_covid19_records(dataset_root)
    assert COVID19_CLASS_TO_INDEX == {
        "COVID": 0,
        "Lung_Opacity": 1,
        "Normal": 2,
        "Viral_Pneumonia": 3,
    }
    assert len(records) == 12
    assert Counter(label for _path, label in records) == Counter(
        {0: 3, 1: 3, 2: 3, 3: 3}
    )
    assert all("masks" not in Path(path).parts for path, _label in records)


def test_covid19_stratified_split_is_deterministic_and_disjoint() -> None:
    labels = torch.tensor([0] * 11 + [1] * 13 + [2] * 17 + [3] * 19)
    first_train, first_test = _stratified_train_test_indices(
        labels, num_classes=4, seed=42
    )
    second_train, second_test = _stratified_train_test_indices(
        labels, num_classes=4, seed=42
    )
    other_train, _other_test = _stratified_train_test_indices(
        labels, num_classes=4, seed=43
    )

    assert (first_train, first_test) == (second_train, second_test)
    assert first_train != other_train
    assert set(first_train).isdisjoint(first_test)
    assert set(first_train) | set(first_test) == set(range(len(labels)))
    assert Counter(int(labels[index]) for index in first_train) == Counter(
        {0: 8, 1: 10, 2: 13, 3: 15}
    )
    assert Counter(int(labels[index]) for index in first_test) == Counter(
        {0: 3, 1: 3, 2: 4, 3: 4}
    )


def test_covid19_loaders_use_80_20_split_and_withhold_50(tmp_path: Path) -> None:
    _write_fixture(tmp_path, samples_per_class=20)
    config = FedConfig(
        data_root=str(tmp_path),
        num_clients=2,
        num_benign=2,
        server_validation_size=50,
        batch_size=4,
        num_workers=0,
        dirichlet_alpha=None,
        seed=7,
        device="cpu",
    )
    task = TASK_REGISTRY["covid19"]()
    client_loaders, validation_loader, test_loader = task.build_dataloaders(config)

    assert len(client_loaders) == 2
    assert sum(len(loader.dataset) for loader in client_loaders) == 14
    assert len(validation_loader.dataset) == 50
    assert len(test_loader.dataset) == 16
    validation_indices = set(validation_loader.dataset.indices)
    client_indices = {
        index
        for loader in client_loaders
        for index in loader.dataset.indices
    }
    assert validation_indices.isdisjoint(client_indices)
    assert validation_indices | client_indices == set(range(64))

    images, labels = next(iter(test_loader))
    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)
    assert set(labels.tolist()).issubset({0, 1, 2, 3})


def test_covid19_test_split_is_fixed_across_experiment_seeds(tmp_path: Path) -> None:
    _write_fixture(tmp_path, samples_per_class=20)
    task = TASK_REGISTRY["covid19"]()
    test_records = []
    for seed in (42, 43):
        config = FedConfig(
            data_root=str(tmp_path),
            num_clients=2,
            num_benign=2,
            server_validation_size=8,
            batch_size=4,
            num_workers=0,
            dirichlet_alpha=None,
            seed=seed,
            device="cpu",
        )
        _clients, _validation, test_loader = task.build_dataloaders(config)
        test_records.append(test_loader.dataset.records)
    assert test_records[0] == test_records[1]
