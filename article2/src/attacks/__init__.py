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
from .feddmc_backdoor import (
    FEDDMC_TARGET_LABEL,
    apply_feddmc_trigger,
    evaluate_feddmc_backdoor_asr,
)
from .gaussian import GaussianNoiseAttack
from .label_flipping import LabelFlippingAttack
from .lie import LieAttack, apply_lie_round, lie_parameters, rewrite_lie_uploads
from .lit import (
    LitAttack,
    apply_lit_round,
    lit_attack_metadata,
    lit_parameters,
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
from .scaling import ScalingAttack, scaling_attack_metadata
from .sign_flipping import SignFlippingAttack

__all__ = [
    "ATTACK_CONFIG_VALIDATORS",
    "ATTACK_EVALUATORS",
    "ATTACK_METADATA_BUILDERS",
    "ATTACK_REGISTRY",
    "ROUND_ATTACK_HOOKS",
    "BackdoorAttack",
    "FEDDMC_TARGET_LABEL",
    "GaussianNoiseAttack",
    "LabelFlippingAttack",
    "LieAttack",
    "LitAttack",
    "MaliciousClient",
    "MinMaxAttack",
    "MinSumAttack",
    "MixedAttack",
    "ScalingAttack",
    "SignFlippingAttack",
    "apply_feddmc_trigger",
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
    "evaluate_feddmc_backdoor_asr",
    "evaluate_mixed_attack",
    "lit_attack_metadata",
    "lit_parameters",
    "mixed_attack_for_client",
    "mixed_attack_ids",
    "mixed_attack_metadata",
    "lie_parameters",
    "rewrite_lie_uploads",
    "rewrite_lit_uploads",
    "rewrite_minmax_uploads",
    "rewrite_minsum_uploads",
    "scaling_attack_metadata",
    "validate_attack_config",
]
