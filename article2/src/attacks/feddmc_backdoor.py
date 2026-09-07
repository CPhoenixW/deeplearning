"""Shared primitives copied from the FedDMC reference backdoor experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..config import FedConfig


def apply_feddmc_trigger(inputs: torch.Tensor) -> torch.Tensor:
    """Apply the fixed FedDMC trigger pattern to an image batch.

    The source implementation uses a 3x3 motif in the upper-right area of
    28x28 single-channel images and a nine-pixel motif in the same location of
    every channel of 32x32 RGB images.  The assignments below are the
    vectorized equivalent and intentionally do not reuse the project's generic
    lower-right square trigger.
    """

    if inputs.ndim != 4:
        raise ValueError("FedDMC LIT/Scaling attacks require image inputs.")
    poisoned = inputs.clone()
    channels, height, width = poisoned.shape[1:]
    if channels == 1:
        coordinates = ((1, 24), (1, 25), (1, 26), (2, 24), (3, 25), (4, 26), (5, 24), (5, 25), (5, 26))
        if height <= 5 or width <= 26:
            raise ValueError("FedDMC grayscale trigger requires images at least 6x27.")
        for row, column in coordinates:
            poisoned[:, 0, row, column] = 1.0
        return poisoned
    if channels == 3:
        coordinates = ((1, 28), (1, 29), (1, 30), (2, 29), (3, 28), (4, 29), (5, 28), (5, 29), (5, 30))
        if height <= 5 or width <= 30:
            raise ValueError("FedDMC RGB trigger requires images at least 6x31.")
        for row, column in coordinates:
            poisoned[:, :, row, column] = 1.0
        return poisoned
    raise ValueError(
        "FedDMC reference trigger is defined only for one- or three-channel images."
    )


def evaluate_feddmc_backdoor_asr(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Optional[float]:
    """Measure the reference ASR, excluding inputs already in the target class."""

    model.eval()
    target = int(config.feddmc_backdoor_target_label)
    total = 0
    success = 0
    with torch.no_grad():
        for inputs, labels in loader:
            if inputs.ndim != 4:
                return None
            inputs = apply_feddmc_trigger(inputs.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            eligible = labels != target
            if not bool(eligible.any().item()):
                continue
            if config.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(config.use_amp and device.type == "cuda"),
            ):
                predictions = torch.argmax(model(inputs), dim=1)
            success += int(((predictions == target) & eligible).sum().item())
            total += int(eligible.sum().item())
    return float(success / total) if total else None


def evaluate_feddmc_backdoor_attack(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Optional[float]]:
    return {"backdoor_asr": evaluate_feddmc_backdoor_asr(config, model, loader, device)}


__all__ = [
    "apply_feddmc_trigger",
    "evaluate_feddmc_backdoor_asr",
    "evaluate_feddmc_backdoor_attack",
]
