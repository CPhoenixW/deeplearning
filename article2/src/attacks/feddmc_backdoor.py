"""Shared FedDMC backdoor trigger and ASR evaluation utilities.

The public FedDMC implementation uses the same sparse fixed-pixel trigger for
its targeted LIT and Scaling attacks.  Keeping that trigger separate from this
project's generic square-trigger ``bd`` attack avoids silently changing the
existing backdoor protocol.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import FedConfig


FEDDMC_TARGET_LABEL = 0

# Exact coordinates from Mu et al.'s public ``Adding_Trigger`` helper.
_RGB_ROWS = (1, 1, 1, 2, 3, 4, 5, 5, 5)
_RGB_COLS = (28, 29, 30, 29, 28, 29, 28, 29, 30)
_GRAY_ROWS = (1, 1, 1, 2, 3, 4, 5, 5, 5)
_GRAY_COLS = (24, 25, 26, 24, 25, 26, 24, 25, 26)


def apply_feddmc_trigger(inputs: Tensor) -> Tensor:
    """Return a clone with the FedDMC sparse trigger applied to every image.

    FedDMC defines one pattern for one-channel 28x28-style images and another
    for three-channel 32x32-style images.  The absolute pixel coordinates are
    intentionally preserved because the goal of these attack IDs is experiment
    reproduction, not a resolution-invariant redesign.
    """

    if inputs.ndim != 4:
        raise ValueError("FedDMC LIT/Scaling attacks require image batches (N,C,H,W).")
    channels = int(inputs.shape[1])
    height = int(inputs.shape[2])
    width = int(inputs.shape[3])
    if channels == 1:
        rows, cols = _GRAY_ROWS, _GRAY_COLS
    elif channels == 3:
        rows, cols = _RGB_ROWS, _RGB_COLS
    else:
        raise ValueError(
            "FedDMC's released trigger is defined only for 1- or 3-channel images; "
            f"received C={channels}."
        )
    if height <= max(rows) or width <= max(cols):
        raise ValueError(
            "Input resolution is too small for the released FedDMC trigger: "
            f"received HxW={height}x{width}."
        )

    poisoned = inputs.clone()
    for row, col in zip(rows, cols):
        poisoned[:, :, row, col] = 1.0
    return poisoned


def poison_feddmc_prefix(
    inputs: Tensor,
    labels: Tensor,
    *,
    count: int,
    target_label: int = FEDDMC_TARGET_LABEL,
) -> tuple[Tensor, Tensor]:
    """Poison the first ``count`` samples, matching the FedDMC training code."""

    batch_size = int(labels.shape[0])
    count = max(0, min(int(count), batch_size))
    if count == 0:
        return inputs, labels
    poisoned_inputs = inputs.clone()
    poisoned_labels = labels.clone()
    poisoned_inputs[:count] = apply_feddmc_trigger(poisoned_inputs[:count])
    poisoned_labels[:count] = int(target_label)
    return poisoned_inputs, poisoned_labels


def evaluate_feddmc_backdoor_asr(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float | None:
    """Evaluate the targeted ASR used for FedDMC's LIT/Scaling attacks."""

    del config  # target label and trigger are fixed by the released protocol.
    model.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for inputs, _labels in loader:
            if inputs.ndim != 4:
                return None
            inputs = apply_feddmc_trigger(inputs).to(device, non_blocking=True)
            logits = model(inputs)
            success += int(
                (torch.argmax(logits, dim=1) == FEDDMC_TARGET_LABEL).sum().item()
            )
            total += int(inputs.shape[0])
    return float(success / total) if total > 0 else None


def evaluate_feddmc_backdoor_attack(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float | None]:
    return {
        "backdoor_asr": evaluate_feddmc_backdoor_asr(
            config, model, loader, device
        )
    }


__all__ = [
    "FEDDMC_TARGET_LABEL",
    "apply_feddmc_trigger",
    "evaluate_feddmc_backdoor_asr",
    "evaluate_feddmc_backdoor_attack",
    "poison_feddmc_prefix",
]
