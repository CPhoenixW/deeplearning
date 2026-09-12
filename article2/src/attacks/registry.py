"""Attack registration, construction and coordinated round hooks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Type

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from ..clients import BaseClient, BenignClient, ModelFactory
from ..config import FedConfig, normalize_attack_name
from .backdoor import BackdoorAttack, evaluate_backdoor_attack
from .distance import (
    MinMaxAttack,
    MinSumAttack,
    apply_minmax_round,
    apply_minsum_round,
    distance_attack_metadata,
    validate_distance_attack_config,
)
from .feddmc_backdoor import evaluate_feddmc_backdoor_attack
from .gaussian import GaussianNoiseAttack
from .label_flipping import LabelFlippingAttack
from .lie import LieAttack, apply_lie_round, lie_attack_metadata, validate_lie_config
from .lit import LitAttack, apply_lit_round, lit_attack_metadata, validate_lit_config
from .mixed import (
    MixedAttack,
    apply_mixed_round,
    evaluate_mixed_attack,
    mixed_attack_ids,
    mixed_attack_metadata,
)
from .scaling import ScalingAttack, scaling_attack_metadata
from .sign_flipping import SignFlippingAttack


AttackClientType = Type[BaseClient]
# LIT needs the concrete client objects for its second targeted local-training
# pass, whereas the other round hooks only need model states.  Keep the public
# hook registry generic and let ``apply_round_attack`` dispatch that extra input.
RoundAttackHook = Callable[..., None]
AttackEvaluator = Callable[
    [FedConfig, nn.Module, DataLoader, torch.device],
    Dict[str, Any],
]
AttackMetadataBuilder = Callable[[FedConfig], Dict[str, Any]]
AttackConfigValidator = Callable[[FedConfig], object]


ATTACK_REGISTRY: Dict[str, AttackClientType] = {
    "none": BenignClient,
    "gn": GaussianNoiseAttack,
    "lf": LabelFlippingAttack,
    "sf": SignFlippingAttack,
    "bd": BackdoorAttack,
    "lie": LieAttack,
    "lit": LitAttack,
    "scaling": ScalingAttack,
    "minmax": MinMaxAttack,
    "minsum": MinSumAttack,
    "mix": MixedAttack,
}

ROUND_ATTACK_HOOKS: Dict[str, RoundAttackHook] = {
    "lie": apply_lie_round,
    "lit": apply_lit_round,
    "minmax": apply_minmax_round,
    "minsum": apply_minsum_round,
    "mix": apply_mixed_round,
}

ATTACK_EVALUATORS: Dict[str, AttackEvaluator] = {
    "bd": evaluate_backdoor_attack,
    "lit": evaluate_feddmc_backdoor_attack,
    "scaling": evaluate_feddmc_backdoor_attack,
    "mix": evaluate_mixed_attack,
}

ATTACK_METADATA_BUILDERS: Dict[str, AttackMetadataBuilder] = {
    "lie": lie_attack_metadata,
    "lit": lit_attack_metadata,
    "scaling": scaling_attack_metadata,
    "minmax": lambda config: distance_attack_metadata(config, "minmax"),
    "minsum": lambda config: distance_attack_metadata(config, "minsum"),
    "mix": mixed_attack_metadata,
}

ATTACK_CONFIG_VALIDATORS: Dict[str, AttackConfigValidator] = {
    "lie": validate_lie_config,
    "lit": validate_lit_config,
    "minmax": validate_distance_attack_config,
    "minsum": validate_distance_attack_config,
    "mix": mixed_attack_ids,
}


def validate_attack_config(attack_name: str, config: FedConfig) -> None:
    """Validate generic registration plus optional attack-specific settings."""

    attack_id = normalize_attack_name(attack_name)
    if attack_id not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack {attack_id!r}; available: {sorted(ATTACK_REGISTRY)}"
        )
    validator = ATTACK_CONFIG_VALIDATORS.get(attack_id)
    if validator is not None:
        validator(config)


def create_attack_client(
    attack_name: str,
    client_id: int,
    device: torch.device,
    config: FedConfig,
    loader: DataLoader,
    model_fn: ModelFactory,
) -> BaseClient:
    attack_id = normalize_attack_name(attack_name)
    validate_attack_config(attack_id, config)
    attack_type = ATTACK_REGISTRY[attack_id]
    return attack_type(client_id, device, config, loader, model_fn)


def apply_round_attack(
    config: FedConfig,
    defense_name: str,
    global_state: Dict[str, Tensor],
    client_states: List[Dict[str, Tensor]],
    parameter_names: Sequence[str] | None = None,
    *,
    clients: Sequence[BaseClient] | None = None,
) -> None:
    """Run the optional coordinated hook for the configured attack."""

    attack_id = normalize_attack_name(config.attack_type)
    validate_attack_config(attack_id, config)
    hook = ROUND_ATTACK_HOOKS.get(attack_id)
    if hook is None:
        return
    if attack_id == "lit":
        hook(
            config,
            defense_name,
            global_state,
            client_states,
            parameter_names,
            clients=clients,
        )
        return
    hook(config, defense_name, global_state, client_states, parameter_names)


def evaluate_attack(
    attack_name: str,
    config: FedConfig,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run optional attack-specific evaluation without branching in the pipeline."""

    attack_id = normalize_attack_name(attack_name)
    validate_attack_config(attack_id, config)
    evaluator = ATTACK_EVALUATORS.get(attack_id)
    return {} if evaluator is None else evaluator(config, model, loader, device)


def attack_metadata(attack_name: str, config: FedConfig) -> Dict[str, Any]:
    """Return optional attack-specific structured-result metadata."""

    attack_id = normalize_attack_name(attack_name)
    validate_attack_config(attack_id, config)
    builder = ATTACK_METADATA_BUILDERS.get(attack_id)
    return {} if builder is None else builder(config)


__all__ = [
    "ATTACK_CONFIG_VALIDATORS",
    "ATTACK_EVALUATORS",
    "ATTACK_METADATA_BUILDERS",
    "ATTACK_REGISTRY",
    "ROUND_ATTACK_HOOKS",
    "apply_round_attack",
    "attack_metadata",
    "create_attack_client",
    "evaluate_attack",
    "validate_attack_config",
]
