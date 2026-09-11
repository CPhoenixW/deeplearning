"""Regression coverage for the fixed layer-aware SVDD descriptor."""

from __future__ import annotations

import hashlib

import torch

from src.fixed_descriptor import FixedLayerDescriptor


def test_layer_descriptor_matches_the_validated_mapping() -> None:
    reference = {
        "layer_a": torch.zeros(2, 3),
        "layer_b": torch.zeros(4),
    }
    client = {
        "layer_a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer_b": torch.tensor([2.0, -3.0, 5.0, -7.0]),
    }

    descriptor = FixedLayerDescriptor(
        reference,
        parameter_names=("layer_a", "layer_b"),
        output_dim=4096,
        seed=2027,
        projection_device="cpu",
    ).describe(client, reference)

    assert descriptor.shape == (4096,)
    assert hashlib.sha256(descriptor.numpy().tobytes()).hexdigest() == (
        "079ae5f40f22d131d41b59803e7ab7097c92bfeb0c37bd5da4b5093565e5c132"
    )


def test_layer_descriptor_uses_all_4096_features_for_layers() -> None:
    reference = {"layer_a": torch.zeros(2), "layer_b": torch.zeros(3)}
    descriptor = FixedLayerDescriptor(
        reference,
        parameter_names=("layer_a", "layer_b"),
        output_dim=4096,
        seed=9,
        projection_device="cpu",
    )

    assert descriptor.layout.layer_dim == 4096
    assert descriptor.layout.global_dim == 0
    assert descriptor._buckets.min().item() >= 0
    assert descriptor._buckets.max().item() < 4096
