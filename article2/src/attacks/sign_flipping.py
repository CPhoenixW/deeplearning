"""Sign-flipping model-poisoning attack."""

from __future__ import annotations

from typing import Dict

from torch import Tensor

from .base import MaliciousClient


class SignFlippingAttack(MaliciousClient):
    """Upload ``global - scale * (local - global)`` after local training."""

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        scale = float(self.config.sign_flip_scale)
        flipped: Dict[str, Tensor] = {}
        for key, global_value in global_state_dict.items():
            global_cpu = global_value.detach().cpu()
            local_cpu = local_state_dict[key].detach().cpu()
            if global_cpu.is_floating_point():
                flipped[key] = (
                    global_cpu - scale * (local_cpu - global_cpu)
                ).clone()
            else:
                flipped[key] = global_cpu.clone()
        return flipped


__all__ = ["SignFlippingAttack"]
