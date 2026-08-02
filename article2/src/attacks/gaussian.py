"""Gaussian model-poisoning attack."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .base import MaliciousClient


class GaussianNoiseAttack(MaliciousClient):
    """Replace every floating tensor with a moment-matched Gaussian draw."""

    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if not self.config.skip_redundant_attack_training:
            return super().local_step(global_state_dict, reference_state_dict)
        reference = reference_state_dict or global_state_dict
        return self._postprocess_upload(reference, reference)

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        scale = float(self.config.gaussian_sigma)
        noisy: Dict[str, Tensor] = {}
        for key, value in global_state_dict.items():
            tensor = value.detach().cpu()
            if tensor.is_floating_point():
                floating = tensor.float()
                mean = floating.mean()
                std = floating.std(unbiased=False).clamp_min(1e-8)
                output = mean + scale * std * torch.randn_like(floating)
                noisy[key] = output.to(dtype=tensor.dtype).clone()
            else:
                noisy[key] = tensor.clone()
        return noisy


__all__ = ["GaussianNoiseAttack"]
