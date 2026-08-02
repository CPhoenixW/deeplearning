"""Tests for modular attack registration and mixed round hooks."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import (
    ATTACK_EVALUATORS,
    ATTACK_METADATA_BUILDERS,
    ATTACK_REGISTRY,
    ROUND_ATTACK_HOOKS,
    MixedAttack,
    apply_round_attack,
    attack_metadata,
    evaluate_attack,
    mixed_attack_for_client,
    validate_attack_config,
)
from src.config import FedConfig


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value], dtype=torch.float32),
        "counter": torch.tensor(0, dtype=torch.long),
    }


def test_every_non_clean_attack_is_implemented_in_attacks_package() -> None:
    assert set(ATTACK_REGISTRY) == {"none", "gn", "lf", "sf", "bd", "lie", "mix"}
    assert set(ROUND_ATTACK_HOOKS) == {"lie", "mix"}
    assert set(ATTACK_EVALUATORS) == {"bd", "mix"}
    assert set(ATTACK_METADATA_BUILDERS) == {"mix"}
    assert all(
        attack_id == "none" or attack.__module__.startswith("src.attacks.")
        for attack_id, attack in ATTACK_REGISTRY.items()
    )


def test_mixed_attack_delegates_deterministically() -> None:
    config = FedConfig(
        num_clients=6,
        num_benign=2,
        mixed_attack_types="label_flipping,lie_attack,gn,bd",
    )
    expected = ["lf", "lie", "gn", "bd"]
    assert [mixed_attack_for_client(config, client_id) for client_id in range(2, 6)] == expected

    for client_id, attack_id in zip(range(2, 6), expected):
        mixed = MixedAttack(
            client_id,
            torch.device("cpu"),
            config,
            [],
            lambda: (_ for _ in ()).throw(AssertionError("unused model")),
        )
        assert mixed.attack_id == attack_id
        assert mixed.delegate.__class__ is ATTACK_REGISTRY[attack_id]


def test_mixed_round_hook_rewrites_only_lie_clients() -> None:
    config = FedConfig(
        num_clients=6,
        num_benign=2,
        attack_type="mix",
        mixed_attack_types="lf,lie,gn,lie",
        lie_z_override=0.0,
    )
    global_state = _state(0.0)
    client_states = [
        _state(1.0),
        _state(3.0),
        _state(10.0),
        _state(11.0),
        _state(12.0),
        _state(13.0),
    ]
    originals = copy.deepcopy(client_states)

    apply_round_attack(config, "svdd", global_state, client_states)

    # LF and GN delegates keep their already-produced upload in this round hook.
    assert torch.equal(client_states[2]["weight"], originals[2]["weight"])
    assert torch.equal(client_states[4]["weight"], originals[4]["weight"])
    # With z=0, coordinated LIE uploads equal the mean benign delta: (1+3)/2=2.
    assert torch.equal(client_states[3]["weight"], torch.tensor([2.0]))
    assert torch.equal(client_states[5]["weight"], torch.tensor([2.0]))


def test_mixed_attack_composes_metadata_validation_and_backdoor_evaluation() -> None:
    class TargetZeroModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((inputs.shape[0], 2), dtype=torch.float32)
            logits[:, 0] = 1.0
            return logits

    config = FedConfig(
        num_clients=3,
        num_benign=1,
        attack_type="mix",
        mixed_attack_types="lf,bd",
        backdoor_target_label=0,
        backdoor_trigger_size=1,
    )
    validate_attack_config("mix", config)
    assert attack_metadata("mix", config)["mixed_attack_assignments"] == {
        "1": "lf",
        "2": "bd",
    }

    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1, 2, 2), torch.ones(2, dtype=torch.long)),
        batch_size=2,
    )
    assert evaluate_attack(
        "mix", config, TargetZeroModel(), loader, torch.device("cpu")
    ) == {"backdoor_asr": 1.0}

    config.num_clients = 2
    assert evaluate_attack(
        "mix", config, TargetZeroModel(), loader, torch.device("cpu")
    ) == {}

    config.mixed_attack_types = "lf,mix"
    try:
        validate_attack_config("mix", config)
    except ValueError as exc:
        assert "nested 'mix'" in str(exc)
    else:
        raise AssertionError("nested mixed attacks must be rejected")
