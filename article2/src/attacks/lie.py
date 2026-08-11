"""Paper-faithful LIE / ALIE coordinated model-poisoning attack.

Reference: Baruch et al., *A Little Is Enough: Circumventing Defenses for
Distributed Learning* (NeurIPS 2019).  The source formulation is over
gradients.  This project sends model deltas, so ``mu + z * sigma`` below is the
sign-equivalent state-delta form of the reference gradient construction
``mu - z * sigma``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from ..config import FedConfig
from .coordinated import (
    CoordinatedModelPoisoningAttack,
    StateDict,
    attack_compute_device,
    attack_parameter_names,
    benign_states,
    empty_crafted_delta,
    iter_benign_delta_chunks,
    rewrite_crafted_uploads,
)


class LieAttack(CoordinatedModelPoisoningAttack):
    """LIE client shell; a coordinated hook replaces its final upload."""


def lie_parameters(
    config: FedConfig,
    *,
    attacker_count: int | None = None,
) -> tuple[int, float]:
    """Return the paper's ``(s, z_max)`` for the configured population.

    The original implementation defines

    ``s = floor(N / 2) + 1 - M`` and
    ``z_max = Phi^-1((N - M - s) / (N - M))``.

    ``lie_z_override`` remains an explicit sensitivity-control escape hatch,
    but all default and generated experiment configurations leave it unset so
    that changing client or attacker counts automatically changes ``z_max``.
    """

    total = int(config.num_clients)
    attackers = (
        int(config.num_clients) - int(config.num_benign)
        if attacker_count is None
        else int(attacker_count)
    )
    if total < 3:
        raise ValueError("LIE requires at least three total clients.")
    if attackers < 1 or attackers >= total:
        raise ValueError("LIE requires 1 <= malicious clients < num_clients.")

    s = total // 2 + 1 - attackers
    benign_count = total - attackers
    cdf_value = float(benign_count - s) / float(benign_count)
    if config.lie_z_override is not None:
        z = float(config.lie_z_override)
        if not math.isfinite(z):
            raise ValueError("lie_z_override must be finite when provided.")
        return s, z
    if not 0.0 < cdf_value < 1.0:
        raise ValueError(
            "The LIE paper formula produced a non-finite quantile; choose a "
            "valid client/attacker ratio or set lie_z_override explicitly."
        )
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    z = float(normal.icdf(torch.tensor(cdf_value, dtype=torch.float64)).item())
    if not math.isfinite(z):
        raise ValueError("LIE z_max is non-finite.")
    return s, z


def validate_lie_config(config: FedConfig) -> None:
    """Validate the paper formula before local clients begin training."""

    lie_parameters(config)


def lie_attack_metadata(config: FedConfig) -> Dict[str, object]:
    """Record the derived paper parameters in every structured result."""

    s, z = lie_parameters(config)
    return {
        "lie_variant": "ALIE (Baruch et al., NeurIPS 2019)",
        "lie_coordinate_space": "model_delta",
        "lie_std_correction": 1,
        "lie_s": int(s),
        "lie_z": float(z),
        "lie_z_override_active": config.lie_z_override is not None,
    }


def rewrite_lie_uploads(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str] | None = None,
) -> None:
    """Replace selected uploads with the original LIE statistical update.

    ``defense_name`` remains in the public hook signature for compatibility;
    unlike the old implementation, it does not alter the attacker strength.
    A fixed attack must have the same definition against every defense.
    """

    del defense_name
    client_ids = tuple(int(client_id) for client_id in malicious_client_ids)
    if not client_ids:
        return
    _s, z = lie_parameters(config)
    names = attack_parameter_names(global_state, parameter_names)
    states = benign_states(config, client_states)
    device = attack_compute_device(config)
    crafted_delta = empty_crafted_delta(global_state, names)

    for name, start, end, deltas in iter_benign_delta_chunks(
        global_state,
        states,
        names,
        device=device,
    ):
        mean = deltas.mean(dim=0)
        # ``torch.std`` in the reference implementation uses Bessel's
        # correction (``unbiased=True``); preserve that detail exactly.
        std = deltas.std(dim=0, unbiased=True)
        crafted_delta[name].reshape(-1)[start:end] = (
            mean + float(z) * std
        ).detach().cpu()

    rewrite_crafted_uploads(
        global_state,
        client_states,
        client_ids,
        names,
        crafted_delta,
    )


def apply_lie_round(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
) -> None:
    rewrite_lie_uploads(
        config,
        defense_name,
        global_state,
        client_states,
        range(config.num_benign, config.num_clients),
        parameter_names,
    )


__all__ = [
    "LieAttack",
    "apply_lie_round",
    "lie_attack_metadata",
    "lie_parameters",
    "rewrite_lie_uploads",
    "validate_lie_config",
]
