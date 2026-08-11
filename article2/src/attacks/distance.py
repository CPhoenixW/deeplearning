"""Min-Max and Min-Sum omniscient model-poisoning attacks.

Both attacks follow Shejwalkar and Houmansadr, *Manipulating the Byzantine:
Optimizing Model Poisoning Attacks and Defenses for Federated Learning* (NDSS
2021), including the reference implementation's deviation choices and binary
search.  The original code works on gradients; this module writes the
sign-equivalent adversarial model delta used by this repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

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


DistanceObjective = Literal["minmax", "minsum"]
DEVIATION_MODES = ("std", "sign", "unit_vec")
_INITIAL_LAMBDA = 10.0
_SEARCH_TOLERANCE = 1e-5


class MinMaxAttack(CoordinatedModelPoisoningAttack):
    """Min-Max client shell; the coordinated hook writes its final upload."""


class MinSumAttack(CoordinatedModelPoisoningAttack):
    """Min-Sum client shell; the coordinated hook writes its final upload."""


@dataclass(frozen=True)
class _DistanceStatistics:
    pairwise_squared: Tensor
    candidate_a: Tensor
    candidate_b: Tensor
    candidate_c: Tensor

    def candidate_squared_distances(self, lambda_value: float) -> Tensor:
        value = float(lambda_value)
        distances = (
            self.candidate_a
            - 2.0 * value * self.candidate_b
            + (value * value) * self.candidate_c
        )
        # Dot-product accumulation may leave a tiny negative residual at zero.
        return distances.clamp_min(0.0)


def distance_attack_deviation(config: FedConfig) -> str:
    """Resolve the source implementation's perturbation-vector choice."""

    mode = str(config.distance_attack_deviation).lower().strip()
    if mode not in DEVIATION_MODES:
        raise ValueError(
            "distance_attack_deviation must be one of "
            f"{list(DEVIATION_MODES)}, got {config.distance_attack_deviation!r}."
        )
    return mode


def validate_distance_attack_config(config: FedConfig) -> None:
    """Validate shared Min-Max / Min-Sum settings before local training."""

    distance_attack_deviation(config)


def distance_attack_metadata(config: FedConfig, objective: DistanceObjective) -> Dict[str, object]:
    """Expose fixed source-algorithm choices in structured result metadata."""

    if objective not in {"minmax", "minsum"}:
        raise ValueError(f"Unknown distance-attack objective {objective!r}.")
    return {
        "distance_attack_variant": "Min-Max" if objective == "minmax" else "Min-Sum",
        "distance_attack_reference": "Shejwalkar & Houmansadr, NDSS 2021",
        "distance_attack_coordinate_space": "model_delta",
        "distance_attack_deviation": distance_attack_deviation(config),
        "distance_attack_initial_lambda": _INITIAL_LAMBDA,
        "distance_attack_search_tolerance": _SEARCH_TOLERANCE,
        "distance_attack_std_correction": 1,
    }


def _mean_norm(
    global_state: StateDict,
    states: Sequence[StateDict],
    parameter_names: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    norm_sq = torch.zeros((), dtype=torch.float32, device=device)
    for _name, _start, _end, deltas in iter_benign_delta_chunks(
        global_state,
        states,
        parameter_names,
        device=device,
    ):
        mean = deltas.mean(dim=0)
        norm_sq = norm_sq + mean.square().sum()
    return norm_sq.sqrt()


def _deviation(
    deltas: Tensor,
    mean: Tensor,
    *,
    mode: str,
    mean_norm: Tensor | None,
) -> Tensor:
    if mode == "std":
        # This is ``torch.std(all_updates, 0)`` in the reference code.
        return deltas.std(dim=0, unbiased=True)
    if mode == "sign":
        return mean.sign()
    assert mode == "unit_vec"
    assert mean_norm is not None
    if float(mean_norm.item()) == 0.0:
        return torch.zeros_like(mean)
    return mean / mean_norm


def _distance_statistics(
    global_state: StateDict,
    states: Sequence[StateDict],
    parameter_names: Sequence[str],
    *,
    mode: str,
    device: torch.device,
    mean_norm: Tensor | None = None,
) -> _DistanceStatistics:
    """Accumulate the source algorithm's distances without a full update stack."""

    count = len(states)
    if count < 2:
        raise ValueError("Min-Max and Min-Sum require at least two benign updates.")
    if mode == "unit_vec" and mean_norm is None:
        mean_norm = _mean_norm(global_state, states, parameter_names, device=device)
    pairwise_squared = torch.zeros((count, count), dtype=torch.float32, device=device)
    candidate_a = torch.zeros(count, dtype=torch.float32, device=device)
    candidate_b = torch.zeros(count, dtype=torch.float32, device=device)
    candidate_c = torch.zeros((), dtype=torch.float32, device=device)

    for _name, _start, _end, deltas in iter_benign_delta_chunks(
        global_state,
        states,
        parameter_names,
        device=device,
    ):
        mean = deltas.mean(dim=0)
        centered = deltas - mean
        deviation = _deviation(
            deltas,
            mean,
            mode=mode,
            mean_norm=mean_norm,
        )

        # ||u_i-u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2 u_i^T u_j.  Accumulating
        # this per parameter chunk is algebraically equivalent to the reference
        # ``torch.cdist``/``torch.norm`` computation but avoids B x D storage.
        squared_norms = deltas.square().sum(dim=1)
        pairwise_squared = pairwise_squared + (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * (deltas @ deltas.T)
        )
        candidate_a = candidate_a + centered.square().sum(dim=1)
        candidate_b = candidate_b + (centered * deviation).sum(dim=1)
        candidate_c = candidate_c + deviation.square().sum()

    return _DistanceStatistics(
        pairwise_squared=pairwise_squared.clamp_min(0.0),
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        candidate_c=candidate_c,
    )


def _largest_feasible_lambda(
    statistics: _DistanceStatistics,
    *,
    objective: DistanceObjective,
) -> float:
    """Reproduce the source Min-Max/Min-Sum lambda binary search."""

    if objective == "minmax":
        threshold = float(statistics.pairwise_squared.max().item())

        def feasible(value: float) -> bool:
            return float(statistics.candidate_squared_distances(value).max().item()) <= threshold

    elif objective == "minsum":
        threshold = float(statistics.pairwise_squared.sum(dim=1).min().item())

        def feasible(value: float) -> bool:
            return float(statistics.candidate_squared_distances(value).sum().item()) <= threshold

    else:
        raise ValueError(f"Unknown distance-attack objective {objective!r}.")

    # Keep the slightly unusual update order of the authors' public reference
    # implementation (initial lambda=10, lambda_fail=10, tolerance=1e-5).
    lambda_value = _INITIAL_LAMBDA
    lambda_fail = _INITIAL_LAMBDA
    lambda_success = 0.0
    for _ in range(64):
        if abs(lambda_success - lambda_value) <= _SEARCH_TOLERANCE:
            break
        if feasible(lambda_value):
            lambda_success = lambda_value
            lambda_value = lambda_value + lambda_fail / 2.0
        else:
            lambda_value = lambda_value - lambda_fail / 2.0
        lambda_fail = lambda_fail / 2.0
    if not math.isfinite(lambda_success):
        raise FloatingPointError("Distance-attack lambda search became non-finite.")
    return float(lambda_success)


def _craft_distance_delta(
    config: FedConfig,
    global_state: StateDict,
    client_states: Sequence[StateDict],
    parameter_names: Sequence[str],
    *,
    objective: DistanceObjective,
) -> Dict[str, Tensor]:
    mode = distance_attack_deviation(config)
    states = benign_states(config, client_states)
    device = attack_compute_device(config)
    mean_norm = (
        _mean_norm(global_state, states, parameter_names, device=device)
        if mode == "unit_vec"
        else None
    )
    statistics = _distance_statistics(
        global_state,
        states,
        parameter_names,
        mode=mode,
        device=device,
        mean_norm=mean_norm,
    )
    lambda_value = _largest_feasible_lambda(statistics, objective=objective)
    crafted_delta = empty_crafted_delta(global_state, parameter_names)

    for name, start, end, deltas in iter_benign_delta_chunks(
        global_state,
        states,
        parameter_names,
        device=device,
    ):
        mean = deltas.mean(dim=0)
        deviation = _deviation(
            deltas,
            mean,
            mode=mode,
            mean_norm=mean_norm,
        )
        # Reference gradient form: mu_g - lambda * deviation_g.  Model deltas
        # satisfy delta = -eta * gradient, hence the sign-equivalent plus here.
        crafted_delta[name].reshape(-1)[start:end] = (
            mean + lambda_value * deviation
        ).detach().cpu()
    return crafted_delta


def _rewrite_distance_uploads(
    config: FedConfig,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str] | None,
    *,
    objective: DistanceObjective,
) -> None:
    client_ids = tuple(int(client_id) for client_id in malicious_client_ids)
    if not client_ids:
        return
    names = attack_parameter_names(global_state, parameter_names)
    crafted_delta = _craft_distance_delta(
        config,
        global_state,
        client_states,
        names,
        objective=objective,
    )
    rewrite_crafted_uploads(
        global_state,
        client_states,
        client_ids,
        names,
        crafted_delta,
    )


def rewrite_minmax_uploads(
    config: FedConfig,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str] | None = None,
) -> None:
    _rewrite_distance_uploads(
        config,
        global_state,
        client_states,
        malicious_client_ids,
        parameter_names,
        objective="minmax",
    )


def rewrite_minsum_uploads(
    config: FedConfig,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str] | None = None,
) -> None:
    _rewrite_distance_uploads(
        config,
        global_state,
        client_states,
        malicious_client_ids,
        parameter_names,
        objective="minsum",
    )


def apply_minmax_round(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
) -> None:
    del defense_name
    rewrite_minmax_uploads(
        config,
        global_state,
        client_states,
        range(config.num_benign, config.num_clients),
        parameter_names,
    )


def apply_minsum_round(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
) -> None:
    del defense_name
    rewrite_minsum_uploads(
        config,
        global_state,
        client_states,
        range(config.num_benign, config.num_clients),
        parameter_names,
    )


__all__ = [
    "DEVIATION_MODES",
    "MinMaxAttack",
    "MinSumAttack",
    "apply_minmax_round",
    "apply_minsum_round",
    "distance_attack_metadata",
    "distance_attack_deviation",
    "rewrite_minmax_uploads",
    "rewrite_minsum_uploads",
    "validate_distance_attack_config",
]
