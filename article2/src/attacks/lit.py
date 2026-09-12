"""Targeted LIT attack used by the released FedDMC experiments.

This is deliberately distinct from ``lie.py``.  The project's existing ``lie``
attack is the paper-faithful ALIE statistical model-poisoning construction from
Baruch et al.  FedDMC's public ``LIT_attack`` uses a related statistical envelope
as part of a *targeted backdoor* procedure:

1. malicious clients first perform ordinary local training;
2. their mean local model is used as the starting point for a second, fully
   backdoored local-training pass on every malicious client;
3. the mean backdoor result is converted to the reference code's candidate
   update and clipped coordinate-wise into ``mu +/- z * sigma``;
4. every malicious client uploads the same crafted model.

The implementation below is algebraically expressed in model-delta space, which
is equivalent to FedDMC's gradient-space code but avoids dividing by the local
learning rate.  ``z`` follows the formula stated in the FedDMC paper and is kept
independent from this project's separate ``lie_z_override`` sensitivity knob.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from ..clients import BaseClient
from ..config import FedConfig
from .base import MaliciousClient
from .coordinated import (
    DELTA_CHUNK_SIZE,
    StateDict,
    attack_compute_device,
    attack_parameter_names,
    empty_crafted_delta,
    rewrite_crafted_uploads,
)
from .feddmc_backdoor import FEDDMC_TARGET_LABEL, poison_feddmc_prefix


class LitAttack(MaliciousClient):
    """Client shell for FedDMC's two-pass targeted LIT construction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lit_backdoor_mode = False

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if not self._lit_backdoor_mode:
            # The first local pass is clean in the released FedDMC code.
            return x, y
        return poison_feddmc_prefix(
            x,
            y,
            count=int(x.shape[0]),
            target_label=FEDDMC_TARGET_LABEL,
        )

    def backdoor_step(self, start_state: StateDict) -> StateDict:
        """Run FedDMC's second local pass from the malicious mean model.

        The reference code adds an MSE term computed from ``Net.state_dict()``.
        PyTorch returns detached tensors from ``state_dict()`` by default, so
        that term does not contribute gradients.  Omitting it here therefore
        reproduces the effective released-code optimization while avoiding a
        misleading no-op loss term.
        """

        self._lit_backdoor_mode = True
        try:
            return super().local_step(start_state, reference_state_dict=start_state)
        finally:
            self._lit_backdoor_mode = False


def lit_parameters(
    config: FedConfig,
    *,
    attacker_count: int | None = None,
) -> tuple[int, float]:
    """Return FedDMC's paper-defined ``(s, z_max)`` for LIT.

    FedDMC states

    ``s = floor(N/2) + 1 - M`` and
    ``z_max = Phi^-1((N-M-s)/(N-M))``.

    This helper intentionally ignores ``config.lie_z_override`` because LIT and
    the standalone ALIE/LIE attack are separate experiment families.
    """

    total = int(config.num_clients)
    attackers = (
        total - int(config.num_benign)
        if attacker_count is None
        else int(attacker_count)
    )
    if total < 3:
        raise ValueError("FedDMC LIT requires at least three total clients.")
    if attackers < 2 or attackers >= total:
        raise ValueError("FedDMC LIT requires 2 <= malicious clients < num_clients.")

    s = total // 2 + 1 - attackers
    benign_count = total - attackers
    cdf_value = float(benign_count - s) / float(benign_count)
    if not 0.0 < cdf_value < 1.0:
        raise ValueError(
            "The FedDMC LIT formula produced an invalid normal quantile for the "
            "configured client/malicious ratio."
        )
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    z = float(normal.icdf(torch.tensor(cdf_value, dtype=torch.float64)).item())
    if not math.isfinite(z):
        raise ValueError("FedDMC LIT z_max is non-finite.")
    return s, z


def _mean_state(
    states: Sequence[StateDict],
    client_ids: Sequence[int],
    global_state: StateDict,
) -> StateDict:
    """Average floating state entries without stacking whole client models."""

    ids = tuple(int(client_id) for client_id in client_ids)
    if not ids:
        raise ValueError("Cannot average an empty client set.")
    output: StateDict = {}
    for name, reference_value in global_state.items():
        reference = reference_value.detach().cpu()
        if not reference.is_floating_point():
            output[name] = reference.clone()
            continue
        acc = torch.zeros_like(reference, dtype=torch.float32, device="cpu")
        for client_id in ids:
            value = states[client_id][name].detach().cpu().float()
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"State shape mismatch for {name!r}: {tuple(value.shape)} != "
                    f"{tuple(reference.shape)}."
                )
            acc.add_(value)
        acc.div_(float(len(ids)))
        output[name] = acc.to(dtype=reference.dtype)
    return output


def _resolve_lit_clients(
    clients: Sequence[BaseClient] | None,
    malicious_client_ids: Sequence[int],
) -> List[LitAttack]:
    if clients is None:
        raise ValueError(
            "FedDMC LIT requires client objects for its second targeted local-training pass."
        )
    resolved: List[LitAttack] = []
    for client_id in malicious_client_ids:
        if not 0 <= int(client_id) < len(clients):
            raise IndexError(f"Malicious client id {client_id} is out of range.")
        client = clients[int(client_id)]
        if not isinstance(client, LitAttack):
            raise TypeError(
                "FedDMC LIT must be run as a standalone 'lit' attack so every "
                "configured malicious client is a LitAttack instance."
            )
        resolved.append(client)
    return resolved


def rewrite_lit_uploads(
    config: FedConfig,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    *,
    clients: Sequence[BaseClient] | None,
    parameter_names: Sequence[str] | None = None,
) -> None:
    """Replace malicious uploads with FedDMC's targeted LIT crafted model."""

    client_ids = tuple(int(client_id) for client_id in malicious_client_ids)
    if not client_ids:
        return
    if len(client_ids) < 2:
        raise ValueError("FedDMC LIT requires at least two malicious clients.")
    lit_clients = _resolve_lit_clients(clients, client_ids)
    names = attack_parameter_names(global_state, parameter_names)
    _s, z = lit_parameters(config, attacker_count=len(client_ids))

    # FedDMC computes the mean of the first-pass malicious local models and
    # starts every targeted second pass from that same model.
    malicious_mean_state = _mean_state(client_states, client_ids, global_state)

    backdoor_states: List[StateDict] = []
    for client in lit_clients:
        backdoor_states.append(client.backdoor_step(malicious_mean_state))
    backdoor_mean_state = _mean_state(
        backdoor_states,
        range(len(backdoor_states)),
        global_state,
    )

    device = attack_compute_device(config)
    crafted_delta = empty_crafted_delta(global_state, names)

    # Reference-code algebra in delta space.  If d_i = W_i-W_g and d_bar is
    # their mean, FedDMC's intermediate ``new_grads`` maps to the candidate
    # delta ``d_backdoor - 2*d_bar``.  The gradient clip
    # ``g_bar +/- z*sigma_g`` maps exactly to ``d_bar +/- z*sigma_d``.
    for name in names:
        reference = global_state[name].detach().cpu().float().reshape(-1)
        backdoor_delta = (
            backdoor_mean_state[name].detach().cpu().float().reshape(-1) - reference
        )
        size = int(reference.numel())
        target = crafted_delta[name].reshape(-1)
        for start in range(0, size, DELTA_CHUNK_SIZE):
            end = min(size, start + DELTA_CHUNK_SIZE)
            ref_chunk = reference[start:end].to(device)
            deltas = torch.stack(
                [
                    client_states[client_id][name]
                    .detach()
                    .cpu()
                    .reshape(-1)[start:end]
                    .to(device=device, dtype=torch.float32)
                    - ref_chunk
                    for client_id in client_ids
                ],
                dim=0,
            )
            if not bool(torch.isfinite(deltas).all().item()):
                raise FloatingPointError(
                    f"Non-finite LIT first-pass update in parameter {name!r}."
                )
            mean = deltas.mean(dim=0)
            std = deltas.std(dim=0, unbiased=True)
            backdoor_chunk = backdoor_delta[start:end].to(device)
            candidate = backdoor_chunk - 2.0 * mean
            low = mean - float(z) * std
            high = mean + float(z) * std
            crafted = torch.maximum(torch.minimum(candidate, high), low)
            target[start:end] = crafted.detach().cpu()

    rewrite_crafted_uploads(
        global_state,
        client_states,
        client_ids,
        names,
        crafted_delta,
    )


def apply_lit_round(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
    *,
    clients: Sequence[BaseClient] | None = None,
) -> None:
    del defense_name
    rewrite_lit_uploads(
        config,
        global_state,
        client_states,
        range(int(config.num_benign), int(config.num_clients)),
        clients=clients,
        parameter_names=parameter_names,
    )


def validate_lit_config(config: FedConfig) -> None:
    malicious = int(config.num_clients) - int(config.num_benign)
    lit_parameters(config, attacker_count=malicious)


def lit_attack_metadata(config: FedConfig) -> Dict[str, object]:
    malicious = int(config.num_clients) - int(config.num_benign)
    s, z = lit_parameters(config, attacker_count=malicious)
    return {
        "lit_variant": "FedDMC targeted LIT_attack",
        "lit_coordinate_space": "model_delta",
        "lit_target_label": FEDDMC_TARGET_LABEL,
        "lit_trigger": "FedDMC sparse fixed-pixel trigger",
        "lit_first_pass": "ordinary malicious local training",
        "lit_second_pass_poison_fraction": 1.0,
        "lit_s": int(s),
        "lit_z": float(z),
        "lit_z_source": "FedDMC paper formula",
        "lit_reference_mse_term_effective": False,
    }


__all__ = [
    "LitAttack",
    "apply_lit_round",
    "lit_attack_metadata",
    "lit_parameters",
    "rewrite_lit_uploads",
    "validate_lit_config",
]
