"""Modular client attack strategies."""

from .backdoor import BackdoorAttack, evaluate_backdoor_asr
from .base import MaliciousClient
from .gaussian import GaussianNoiseAttack
from .label_flipping import LabelFlippingAttack
from .lie import LieAttack, apply_lie_round, rewrite_lie_uploads
from .mixed import (
    MixedAttack,
    apply_mixed_round,
    evaluate_mixed_attack,
    mixed_attack_for_client,
    mixed_attack_ids,
    mixed_attack_metadata,
)
from .registry import (
    ATTACK_CONFIG_VALIDATORS,
    ATTACK_EVALUATORS,
    ATTACK_METADATA_BUILDERS,
    ATTACK_REGISTRY,
    ROUND_ATTACK_HOOKS,
    apply_round_attack,
    attack_metadata,
    create_attack_client,
    evaluate_attack,
    validate_attack_config,
)
from .sign_flipping import SignFlippingAttack

__all__ = [
    "ATTACK_CONFIG_VALIDATORS",
    "ATTACK_EVALUATORS",
    "ATTACK_METADATA_BUILDERS",
    "ATTACK_REGISTRY",
    "ROUND_ATTACK_HOOKS",
    "BackdoorAttack",
    "GaussianNoiseAttack",
    "LabelFlippingAttack",
    "LieAttack",
    "MaliciousClient",
    "MixedAttack",
    "SignFlippingAttack",
    "apply_lie_round",
    "apply_mixed_round",
    "apply_round_attack",
    "attack_metadata",
    "create_attack_client",
    "evaluate_attack",
    "evaluate_backdoor_asr",
    "evaluate_mixed_attack",
    "mixed_attack_for_client",
    "mixed_attack_ids",
    "mixed_attack_metadata",
    "rewrite_lie_uploads",
    "validate_attack_config",
]
