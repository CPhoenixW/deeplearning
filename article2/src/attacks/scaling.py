"""FedDMC Scaling backdoor attack, separate from the generic ``bd`` attack."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..config import FedConfig
from .base import MaliciousClient
from .feddmc_backdoor import apply_feddmc_trigger


class ScalingAttack(MaliciousClient):
    """Poison the first half of each batch and apply the reference scale rule."""

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        poisoned_count = int(y.shape[0]) // 2
        if poisoned_count == 0:
            return x, y
        poisoned_inputs = x.clone()
        poisoned_inputs[:poisoned_count] = apply_feddmc_trigger(
            poisoned_inputs[:poisoned_count]
        )
        poisoned_labels = y.clone()
        poisoned_labels[:poisoned_count] = int(self.config.feddmc_backdoor_target_label)
        return poisoned_inputs, poisoned_labels

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        malicious_count = int(self.config.num_clients) - int(self.config.num_benign)
        if malicious_count < 1:
            raise ValueError("Scaling attack requires at least one malicious client.")
        scale = float(self.config.num_clients) / float(malicious_count) / 2.0
        scaled: Dict[str, Tensor] = {}
        for key, global_value in global_state_dict.items():
            global_cpu = global_value.detach().cpu()
            local_cpu = local_state_dict[key].detach().cpu()
            if global_cpu.is_floating_point():
                scaled[key] = (global_cpu + (local_cpu - global_cpu) * scale).clone()
            else:
                scaled[key] = global_cpu.clone()
        return scaled


def validate_scaling_config(config: FedConfig) -> None:
    """Ensure the configured population actually contains an attacker."""

    if int(config.num_clients) - int(config.num_benign) < 1:
        raise ValueError("Scaling attack requires at least one malicious client.")


def scaling_attack_metadata(config: FedConfig) -> dict[str, object]:
    validate_scaling_config(config)
    malicious_count = int(config.num_clients) - int(config.num_benign)
    return {
        "scaling_variant": "FedDMC reference Scaling_attack",
        "scaling_poisoned_batch_fraction": 0.5,
        "scaling_factor": float(config.num_clients) / float(malicious_count) / 2.0,
        "scaling_target_label": int(config.feddmc_backdoor_target_label),
    }


__all__ = ["ScalingAttack", "scaling_attack_metadata", "validate_scaling_config"]
