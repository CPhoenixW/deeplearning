"""Backdoor data poisoning with optional model replacement."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import FedConfig
from .base import MaliciousClient


class BackdoorAttack(MaliciousClient):
    """Poison a local sample subset and amplify the resulting model update."""

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        poison_ratio = float(self.config.backdoor_poison_ratio)
        target = int(self.config.backdoor_target_label)
        trigger_size = int(self.config.backdoor_trigger_size)
        trigger_value = float(self.config.backdoor_trigger_value)

        if poison_ratio <= 0.0 or trigger_size <= 0:
            return x, y
        mask = torch.rand(y.shape[0], device=self.device) < poison_ratio
        if not bool(mask.any().item()):
            return x, y

        poisoned_labels = y.clone()
        if x.ndim == 4:
            poisoned_inputs = x.clone()
            poisoned_inputs[
                mask, :, -trigger_size:, -trigger_size:
            ] = trigger_value
        else:
            # Text currently uses target-label poisoning without a token trigger.
            poisoned_inputs = x
        poisoned_labels[mask] = target
        return poisoned_inputs, poisoned_labels

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        scale = float(self.config.backdoor_model_replace_scale)
        if scale <= 0.0:
            scale = 1.0

        attacked: Dict[str, Tensor] = {}
        for key, global_value in global_state_dict.items():
            global_cpu = global_value.detach().cpu()
            local_cpu = local_state_dict[key].detach().cpu()
            if global_cpu.is_floating_point():
                attacked[key] = (
                    global_cpu + scale * (local_cpu - global_cpu)
                ).clone()
            else:
                attacked[key] = global_cpu.clone()
        return attacked


def evaluate_backdoor_asr(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Optional[float]:
    """Measure target-label success after applying the configured image trigger."""

    model.eval()
    trigger = int(config.backdoor_trigger_size)
    if trigger <= 0:
        return None
    total = 0
    success = 0
    with torch.no_grad():
        for inputs, _labels in loader:
            if inputs.ndim != 4:
                return None
            inputs = inputs.to(device, non_blocking=True).clone()
            if config.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            inputs[:, :, -trigger:, -trigger:] = float(
                config.backdoor_trigger_value
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(config.use_amp and device.type == "cuda"),
            ):
                logits = model(inputs)
            success += int(
                (
                    torch.argmax(logits, dim=1)
                    == int(config.backdoor_target_label)
                ).sum().item()
            )
            total += int(inputs.shape[0])
    return float(success / total) if total > 0 else None


def evaluate_backdoor_attack(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    """Build backdoor-specific round metrics for the generic pipeline."""

    return {
        "backdoor_asr": evaluate_backdoor_asr(config, model, loader, device)
    }


__all__ = [
    "BackdoorAttack",
    "evaluate_backdoor_asr",
    "evaluate_backdoor_attack",
]
