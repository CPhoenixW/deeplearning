"""LIE/ALIE client placeholder and coordinated round-level rewrite."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from ..config import FedConfig
from .base import MaliciousClient


class LieAttack(MaliciousClient):
    """Skip local SGD when the coordinated round hook will replace the upload."""

    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if not self.config.skip_redundant_attack_training:
            return super().local_step(global_state_dict, reference_state_dict)
        return reference_state_dict or global_state_dict


def _default_lie_s(config: FedConfig, defense_name: str) -> int:
    if defense_name == "mk":
        return int(
            config.krum_num_byzantine
            if config.krum_num_byzantine is not None
            else max(0, config.num_clients - config.num_benign)
        )
    if defense_name == "tm":
        if config.trimmed_mean_num_byzantine is not None:
            return int(config.trimmed_mean_num_byzantine)
        return max(0, config.num_clients - config.num_benign)
    return 0


def rewrite_lie_uploads(
    config: FedConfig,
    defense_name: str,
    global_state: Dict[str, Tensor],
    client_states: List[Dict[str, Tensor]],
    malicious_client_ids: Sequence[int],
) -> None:
    """Rewrite selected uploads as ``mu + z * sigma`` benign-update estimates."""

    client_ids = [int(client_id) for client_id in malicious_client_ids]
    if not client_ids:
        return
    benign_count = int(config.num_benign)
    if benign_count < 1:
        raise ValueError("LIE requires at least one benign client update")

    s = (
        int(config.lie_s)
        if config.lie_s is not None
        else _default_lie_s(config, defense_name)
    )
    s = max(0, min(s, benign_count - 1))
    ratio = float(benign_count - s) / float(benign_count)
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    if config.lie_z_override is not None:
        z = float(config.lie_z_override)
    else:
        z = float(
            torch.distributions.Normal(0.0, 1.0)
            .icdf(torch.tensor(ratio))
            .item()
        )

    benign_states = client_states[:benign_count]
    # Client states are kept on CPU between stages.  The per-layer LIE
    # statistics are large enough to dominate the remaining confirmation run,
    # so use the aggregation device when CUDA aggregation is enabled and move
    # only the compact crafted result back to CPU.
    compute_device = (
        torch.device("cuda")
        if bool(getattr(config, "cuda_aggregation", False))
        and torch.cuda.is_available()
        else torch.device("cpu")
    )
    crafted_delta: Dict[str, Tensor] = {}
    for key, global_value in global_state.items():
        global_cpu = global_value.detach().cpu()
        if not global_cpu.is_floating_point():
            continue
        global_work = global_cpu.float().to(compute_device)
        deltas = torch.stack(
            [
                state[key].detach().to(compute_device, non_blocking=True).float()
                - global_work
                for state in benign_states
            ],
            dim=0,
        )
        mean = deltas.mean(dim=0)
        std = deltas.std(dim=0, unbiased=False)
        crafted_delta[key] = (mean + z * std).detach().cpu()

    for client_id in client_ids:
        if not 0 <= client_id < len(client_states):
            raise IndexError(f"LIE client id {client_id} is out of range")
        source = client_states[client_id]
        rewritten: Dict[str, Tensor] = {}
        for key, global_value in global_state.items():
            global_cpu = global_value.detach().cpu()
            if global_cpu.is_floating_point():
                output = global_cpu.float() + crafted_delta[key]
                rewritten[key] = output.to(dtype=global_cpu.dtype).clone()
            else:
                rewritten[key] = source[key].detach().cpu().clone()
        client_states[client_id] = rewritten


def apply_lie_round(
    config: FedConfig,
    defense_name: str,
    global_state: Dict[str, Tensor],
    client_states: List[Dict[str, Tensor]],
) -> None:
    rewrite_lie_uploads(
        config,
        defense_name,
        global_state,
        client_states,
        range(config.num_benign, config.num_clients),
    )


__all__ = ["LieAttack", "apply_lie_round", "rewrite_lie_uploads"]
