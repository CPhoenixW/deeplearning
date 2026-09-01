"""Modular client attack strategies."""

from .backdoor import BackdoorAttack, evaluate_backdoor_asr
from .base import MaliciousClient
from .distance import (
    MinMaxAttack,
    MinSumAttack,
    apply_minmax_round,
    apply_minsum_round,
    rewrite_minmax_uploads,
    rewrite_minsum_uploads,
)
from .gaussian import GaussianNoiseAttack
from .label_flipping import LabelFlippingAttack
from .lie import LieAttack, apply_lie_round, lie_parameters, rewrite_lie_uploads
from .lit import (
    FEDDMC_LIT_POISON_RATIO,
    FEDDMC_LIT_Z,
    LitAttack,
    apply_lit_round,
    evaluate_lit_attack,
    lit_attack_metadata,
    rewrite_lit_uploads,
)
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
    "FEDDMC_LIT_POISON_RATIO",
    "FEDDMC_LIT_Z",
    "GaussianNoiseAttack",
    "LabelFlippingAttack",
    "LieAttack",
    "LitAttack",
    "MaliciousClient",
    "MinMaxAttack",
    "MinSumAttack",
    "MixedAttack",
    "SignFlippingAttack",
    "apply_lie_round",
    "apply_lit_round",
    "apply_minmax_round",
    "apply_minsum_round",
    "apply_mixed_round",
    "apply_round_attack",
    "attack_metadata",
    "create_attack_client",
    "evaluate_attack",
    "evaluate_backdoor_asr",
    "evaluate_lit_attack",
    "evaluate_mixed_attack",
    "lit_attack_metadata",
    "mixed_attack_for_client",
    "mixed_attack_ids",
    "mixed_attack_metadata",
    "lie_parameters",
    "rewrite_lie_uploads",
    "rewrite_lit_uploads",
    "rewrite_minmax_uploads",
    "rewrite_minsum_uploads",
    "validate_attack_config",
]
