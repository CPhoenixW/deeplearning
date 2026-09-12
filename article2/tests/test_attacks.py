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
    lie_parameters,
    mixed_attack_for_client,
    rewrite_minmax_uploads,
    rewrite_minsum_uploads,
    validate_attack_config,
)
from src.config import FedConfig


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value], dtype=torch.float32),
        "counter": torch.tensor(0, dtype=torch.long),
    }


def _parameter_state(value: torch.Tensor, running: float = -1.0) -> dict[str, torch.Tensor]:
    return {
        "weight": value.detach().clone().float(),
        "running_mean": torch.tensor([running], dtype=torch.float32),
        "counter": torch.tensor(0, dtype=torch.long),
    }


def test_every_non_clean_attack_is_implemented_in_attacks_package() -> None:
    assert set(ATTACK_REGISTRY) == {
        "none",
        "gn",
        "lf",
        "sf",
        "bd",
        "lie",
        "lit",
        "minmax",
        "minsum",
        "mix",
    }
    assert set(ROUND_ATTACK_HOOKS) == {"lie", "lit", "minmax", "minsum", "mix"}
    assert set(ATTACK_EVALUATORS) == {"bd", "lit", "mix"}
    assert set(ATTACK_METADATA_BUILDERS) == {"lie", "lit", "minmax", "minsum", "mix"}
    assert all(
        attack_id == "none" or attack.__module__.startswith("src.attacks.")
        for attack_id, attack in ATTACK_REGISTRY.items()
    )


def test_mixed_attack_delegates_deterministically() -> None:
    config = FedConfig(
        num_clients=6,
        num_benign=2,
        mixed_attack_types="label_flipping,lie_attack,gn,bd",
        lie_z_override=0.0,
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


def test_lie_uses_the_paper_formula_sample_std_and_parameter_boundary() -> None:
    config = FedConfig(num_clients=10, num_benign=7, attack_type="lie")
    s, z = lie_parameters(config)
    expected_z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(4.0 / 7.0))
    assert s == 3
    assert abs(z - float(expected_z.item())) < 1e-6

    global_state = _parameter_state(torch.zeros(1), running=17.0)
    benign_values = torch.arange(7, dtype=torch.float32).unsqueeze(1)
    client_states = [
        _parameter_state(value, running=float(index))
        for index, value in enumerate(benign_values)
    ] + [
        _parameter_state(torch.tensor([50.0]), running=50.0),
        _parameter_state(torch.tensor([60.0]), running=60.0),
        _parameter_state(torch.tensor([70.0]), running=70.0),
    ]
    alternative_defense_states = copy.deepcopy(client_states)

    apply_round_attack(
        config,
        "avg",
        global_state,
        client_states,
        parameter_names=("weight",),
    )
    apply_round_attack(
        config,
        "tm",
        global_state,
        alternative_defense_states,
        parameter_names=("weight",),
    )

    expected_delta = benign_values.mean(dim=0) + z * benign_values.std(
        dim=0, unbiased=True
    )
    for index in range(7, 10):
        assert torch.allclose(client_states[index]["weight"], expected_delta)
        assert torch.equal(client_states[index]["weight"], alternative_defense_states[index]["weight"])
        # ALIE is defined over trainable gradient coordinates; BN buffers are
        # held at the global state rather than carrying a malicious local value.
        assert torch.equal(client_states[index]["running_mean"], global_state["running_mean"])
        assert torch.equal(client_states[index]["counter"], global_state["counter"])


def test_minmax_and_minsum_match_their_benign_distance_constraints() -> None:
    global_state = _parameter_state(torch.zeros(2), running=9.0)
    benign_updates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
        dtype=torch.float32,
    )
    config = FedConfig(
        num_clients=6,
        num_benign=4,
        distance_attack_deviation="std",
    )

    for rewrite, objective in (
        (rewrite_minmax_uploads, "minmax"),
        (rewrite_minsum_uploads, "minsum"),
    ):
        client_states = [
            _parameter_state(update, running=float(index))
            for index, update in enumerate(benign_updates)
        ] + [
            _parameter_state(torch.tensor([10.0, 10.0]), running=10.0),
            _parameter_state(torch.tensor([20.0, 20.0]), running=20.0),
        ]
        rewrite(
            config,
            global_state,
            client_states,
            (4, 5),
            parameter_names=("weight",),
        )
        crafted = client_states[4]["weight"]
        assert torch.equal(crafted, client_states[5]["weight"])
        assert not torch.allclose(crafted, benign_updates.mean(dim=0))
        pairwise = torch.cdist(benign_updates, benign_updates).square()
        crafted_distances = torch.cdist(crafted.unsqueeze(0), benign_updates).square()
        if objective == "minmax":
            assert crafted_distances.max().item() <= pairwise.max().item() + 1e-4
        else:
            assert crafted_distances.sum().item() <= pairwise.sum(dim=1).min().item() + 1e-4
        for index in (4, 5):
            assert torch.equal(client_states[index]["running_mean"], global_state["running_mean"])
            assert torch.equal(client_states[index]["counter"], global_state["counter"])


def test_minmax_and_minsum_are_sign_equivalent_to_the_reference_gradient_code() -> None:
    def reference_gradient_attack(
        benign_gradients: torch.Tensor,
        objective: str,
    ) -> torch.Tensor:
        """Literal scalar/vector form of the cited public implementation."""

        mean = benign_gradients.mean(dim=0)
        deviation = benign_gradients.std(dim=0, unbiased=True)
        pairwise = torch.cdist(benign_gradients, benign_gradients).square()
        threshold = (
            pairwise.max()
            if objective == "minmax"
            else pairwise.sum(dim=1).min()
        )
        lambda_value = torch.tensor(10.0)
        lambda_fail = lambda_value.clone()
        lambda_success = torch.tensor(0.0)
        while torch.abs(lambda_success - lambda_value) > 1e-5:
            candidate = mean - lambda_value * deviation
            distance = torch.cdist(candidate.unsqueeze(0), benign_gradients).square()
            score = distance.max() if objective == "minmax" else distance.sum()
            if score <= threshold:
                lambda_success = lambda_value
                lambda_value = lambda_value + lambda_fail / 2.0
            else:
                lambda_value = lambda_value - lambda_fail / 2.0
            lambda_fail = lambda_fail / 2.0
        return mean - lambda_success * deviation

    benign_gradients = torch.tensor(
        [[1.0, -2.0, 0.5], [2.5, -1.0, 1.2], [0.1, -3.0, 2.0], [1.5, -0.2, -0.5]],
        dtype=torch.float32,
    )
    global_state = _parameter_state(torch.zeros(3))
    config = FedConfig(num_clients=6, num_benign=4, distance_attack_deviation="std")
    # The pipeline sends model deltas, d=-g, rather than gradients g.
    benign_deltas = -benign_gradients
    for rewrite, objective in (
        (rewrite_minmax_uploads, "minmax"),
        (rewrite_minsum_uploads, "minsum"),
    ):
        client_states = [
            _parameter_state(delta) for delta in benign_deltas
        ] + [
            _parameter_state(torch.full((3,), 9.0)),
            _parameter_state(torch.full((3,), 8.0)),
        ]
        rewrite(
            config,
            global_state,
            client_states,
            (4, 5),
            parameter_names=("weight",),
        )
        assert torch.allclose(
            client_states[4]["weight"],
            -reference_gradient_attack(benign_gradients, objective),
            atol=5e-4,
            rtol=0.0,
        )


def test_distance_attacks_keep_large_finite_updates_finite() -> None:
    config = FedConfig(num_clients=4, num_benign=2, distance_attack_deviation="std")
    global_state = _parameter_state(torch.zeros(4))
    client_states = [
        _parameter_state(torch.full((4,), 1e20)),
        _parameter_state(torch.full((4,), -1e20)),
        _parameter_state(torch.zeros(4)),
        _parameter_state(torch.zeros(4)),
    ]

    for rewrite in (rewrite_minmax_uploads, rewrite_minsum_uploads):
        states = copy.deepcopy(client_states)
        rewrite(config, global_state, states, (2, 3), parameter_names=("weight",))
        assert all(torch.isfinite(state["weight"]).all() for state in states)


def test_mixed_round_hook_composes_lie_minmax_and_minsum() -> None:
    config = FedConfig(
        num_clients=8,
        num_benign=3,
        attack_type="mix",
        mixed_attack_types="min-max,min_sum,lie,lf,gn",
        lie_z_override=0.0,
    )
    global_state = _parameter_state(torch.zeros(2))
    client_states = [
        _parameter_state(torch.tensor([float(index), float(index + 1)]))
        for index in range(8)
    ]
    originals = copy.deepcopy(client_states)

    apply_round_attack(
        config,
        "svdd",
        global_state,
        client_states,
        parameter_names=("weight",),
    )

    # IDs 3, 4, and 5 receive Min-Max, Min-Sum, and LIE, respectively.
    for index in (3, 4, 5):
        assert not torch.equal(client_states[index]["weight"], originals[index]["weight"])
    # Data/noise delegates already produced their own uploads; this coordinated
    # stage must not overwrite their state.
    for index in (6, 7):
        assert torch.equal(client_states[index]["weight"], originals[index]["weight"])


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
