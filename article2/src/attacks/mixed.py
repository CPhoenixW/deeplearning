"""Deterministic simultaneous mixed-attack composition."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..clients import BaseClient, ModelFactory
from ..config import FedConfig, normalize_attack_name
from .distance import (
    distance_attack_metadata,
    rewrite_minmax_uploads,
    rewrite_minsum_uploads,
    validate_distance_attack_config,
)
from .lie import lie_attack_metadata, rewrite_lie_uploads, validate_lie_config


def mixed_attack_ids(config: FedConfig) -> Tuple[str, ...]:
    """Validate and return the ordered attack family used by a mixed run."""

    raw = str(
        getattr(config, "mixed_attack_types", "lf,bd,gn,sf,lie,minmax,minsum")
    )
    attack_ids = tuple(
        normalize_attack_name(item)
        for item in raw.split(",")
        if item.strip()
    )
    if not attack_ids:
        raise ValueError("mixed_attack_types must contain at least one attack id")
    if any(attack_id in {"none", "mix"} for attack_id in attack_ids):
        raise ValueError("mixed_attack_types cannot contain 'none' or nested 'mix'")

    # Local import avoids a registry/module initialization cycle.
    from .registry import ATTACK_REGISTRY

    unknown = [attack_id for attack_id in attack_ids if attack_id not in ATTACK_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown mixed attacks {unknown}; available: {sorted(ATTACK_REGISTRY)}"
        )
    if "lie" in attack_ids:
        validate_lie_config(config)
    if any(attack_id in {"minmax", "minsum"} for attack_id in attack_ids):
        validate_distance_attack_config(config)
    return attack_ids


def mixed_attack_for_client(config: FedConfig, client_id: int) -> str:
    """Assign attack families round-robin by malicious-client index."""

    if not int(config.num_benign) <= int(client_id) < int(config.num_clients):
        raise ValueError(f"Client {client_id} is not a configured malicious client")
    attack_ids = mixed_attack_ids(config)
    offset = int(client_id) - int(config.num_benign)
    return attack_ids[offset % len(attack_ids)]


class MixedAttack(BaseClient):
    """Delegate each malicious client to one configured attack family."""

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        config: FedConfig,
        loader: DataLoader,
        model_fn: ModelFactory,
    ) -> None:
        super().__init__(client_id, device, config, loader)
        self.attack_id = mixed_attack_for_client(config, client_id)

        from .registry import create_attack_client

        self.delegate = create_attack_client(
            self.attack_id,
            client_id,
            device,
            config,
            loader,
            model_fn,
        )

    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return self.delegate.local_step(global_state_dict, reference_state_dict)


def apply_mixed_round(
    config: FedConfig,
    defense_name: str,
    global_state: Dict[str, Tensor],
    client_states: List[Dict[str, Tensor]],
    parameter_names: Sequence[str] | None = None,
) -> None:
    """Apply coordinated hooks required by attacks inside the mixture."""

    client_ids_by_attack: Dict[str, List[int]] = {}
    for client_id in range(config.num_benign, config.num_clients):
        attack_id = mixed_attack_for_client(config, client_id)
        client_ids_by_attack.setdefault(attack_id, []).append(client_id)

    # Each omniscient component is constructed from the same benign-update
    # view, then rewrites only the malicious clients assigned to that family.
    # The order is immaterial because the client-ID groups are disjoint.
    rewrite_lie_uploads(
        config,
        defense_name,
        global_state,
        client_states,
        client_ids_by_attack.get("lie", ()),
        parameter_names,
    )
    rewrite_minmax_uploads(
        config,
        global_state,
        client_states,
        client_ids_by_attack.get("minmax", ()),
        parameter_names,
    )
    rewrite_minsum_uploads(
        config,
        global_state,
        client_states,
        client_ids_by_attack.get("minsum", ()),
        parameter_names,
    )


def mixed_attack_metadata(config: FedConfig) -> Dict[str, Any]:
    """Describe the effective per-client attack assignment for result files."""

    assignments = {
        str(client_id): mixed_attack_for_client(config, client_id)
        for client_id in range(config.num_benign, config.num_clients)
    }
    counts = Counter(assignments.values())
    component_metadata: Dict[str, Any] = {}
    if "lie" in counts:
        component_metadata["lie"] = lie_attack_metadata(config)
    for attack_id in ("minmax", "minsum"):
        if attack_id in counts:
            component_metadata[attack_id] = distance_attack_metadata(config, attack_id)
    return {
        "mixed_attack_types": str(config.mixed_attack_types),
        "mixed_attack_assignments": assignments,
        "mixed_attack_counts": dict(sorted(counts.items())),
        "mixed_attack_component_metadata": component_metadata,
    }


def evaluate_mixed_attack(
    config: FedConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Compose evaluation hooks for attack families actually assigned this run."""

    from .registry import evaluate_attack

    active_attack_ids = dict.fromkeys(
        mixed_attack_for_client(config, client_id)
        for client_id in range(config.num_benign, config.num_clients)
    )
    metrics: Dict[str, Any] = {}
    for attack_id in active_attack_ids:
        metrics.update(evaluate_attack(attack_id, config, model, loader, device))
    return metrics


__all__ = [
    "MixedAttack",
    "apply_mixed_round",
    "evaluate_mixed_attack",
    "mixed_attack_for_client",
    "mixed_attack_ids",
    "mixed_attack_metadata",
]
