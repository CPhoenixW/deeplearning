"""FedDMC Scaling attack reproduction.

Mu et al.'s released code poisons the first half of every malicious client's
local mini-batch with a fixed sparse trigger targeting class 0, then applies
model replacement with

    scale = (N / M) / 2 = N / (2M),

where ``N`` is the number of participating clients and ``M`` is the number of
malicious clients.  This attack is intentionally separate from the project's
generic ``bd`` attack, whose poison ratio, trigger, and replacement scale are
configurable and therefore define a different experimental protocol.
"""

from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

from ..config import FedConfig
from .base import MaliciousClient
from .feddmc_backdoor import FEDDMC_TARGET_LABEL, poison_feddmc_prefix


class ScalingAttack(MaliciousClient):
    """Backdoor local training followed by FedDMC model scaling."""

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # The reference implementation deterministically poisons the first half
        # of each batch rather than sampling a Bernoulli mask.
        return poison_feddmc_prefix(
            x,
            y,
            count=int(x.shape[0]) // 2,
            target_label=FEDDMC_TARGET_LABEL,
        )

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        total = int(self.config.num_clients)
        malicious = total - int(self.config.num_benign)
        if malicious <= 0:
            raise ValueError("FedDMC Scaling requires at least one malicious client.")
        scale = float(total) / (2.0 * float(malicious))

        attacked: Dict[str, Tensor] = {}
        for key, global_value in global_state_dict.items():
            global_cpu = global_value.detach().cpu()
            local_cpu = local_state_dict[key].detach().cpu()
            if global_cpu.is_floating_point():
                attacked[key] = (
                    global_cpu.float()
                    + scale * (local_cpu.float() - global_cpu.float())
                ).to(dtype=global_cpu.dtype)
            else:
                # Keep integer bookkeeping buffers valid.  The released code
                # flattens state_dict tensors together, but integer BN counters
                # are not meaningful model-replacement coordinates.
                attacked[key] = global_cpu.clone()
        return attacked


def scaling_attack_metadata(config: FedConfig) -> Dict[str, object]:
    total = int(config.num_clients)
    malicious = total - int(config.num_benign)
    if malicious <= 0:
        scale = None
    else:
        scale = float(total) / (2.0 * float(malicious))
    return {
        "scaling_variant": "FedDMC released Scaling_attack",
        "scaling_target_label": FEDDMC_TARGET_LABEL,
        "scaling_batch_poison_fraction": 0.5,
        "scaling_model_replace_scale": scale,
        "scaling_trigger": "FedDMC sparse fixed-pixel trigger",
    }


__all__ = ["ScalingAttack", "scaling_attack_metadata"]
