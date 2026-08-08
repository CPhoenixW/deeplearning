from __future__ import annotations

import torch
from torch import nn

from src.attacks import GaussianNoiseAttack, LieAttack
from src.config import (
    FedConfig,
    load_fed_config_values,
    load_hyperparameter_table,
    resolve_hyperparameters,
)
from src.defenses.base import BaseDefense
from src.utils import (
    aggregate_trimmed_mean,
    clip_client_updates,
    compute_multi_krum_scores,
    weighted_fedavg,
)


def _tiny_state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value, value + 1.0]),
        "num_batches_tracked": torch.tensor(3, dtype=torch.long),
    }


def _unused_model() -> nn.Module:
    raise AssertionError("attack must not build or train a local model")


def test_federated_config_enables_fast_path() -> None:
    values = load_fed_config_values("configs/federated.json")
    hyperparameters = resolve_hyperparameters(
        load_hyperparameter_table("configs/hyperparameters.json"),
        "gn",
        "svdd",
        "cifar10",
    )
    # On RTX 4060 + 32x32 CIFAR batches, AMP/channels-last are slower; keep
    # them available as opt-ins for larger workloads.
    assert not values["use_amp"]
    assert not values["channels_last"]
    assert values["cuda_aggregation"]
    assert values["reuse_client_model"]
    assert hyperparameters["skip_redundant_attack_training"]
    assert not values["round_diagnostics"]


def test_gaussian_and_lie_skip_discarded_local_training() -> None:
    cfg = FedConfig(
        num_clients=2,
        num_benign=1,
        gaussian_sigma=0.0,
        skip_redundant_attack_training=True,
    )
    reference = _tiny_state(2.0)
    loader = []

    gaussian = GaussianNoiseAttack(1, torch.device("cpu"), cfg, loader, _unused_model)
    gaussian_upload = gaussian.local_step(reference)
    assert gaussian_upload.keys() == reference.keys()

    lie = LieAttack(1, torch.device("cpu"), cfg, loader, _unused_model)
    assert lie.local_step(reference) is reference


def test_aggregation_device_keeps_results_equivalent_on_cpu() -> None:
    states = [_tiny_state(1.0), _tiny_state(2.0), _tiny_state(10.0)]
    alpha = torch.tensor([0.25, 0.5, 0.25])

    implicit = weighted_fedavg(states, alpha)
    explicit = weighted_fedavg(states, alpha, device="cpu")
    assert torch.equal(implicit["weight"], explicit["weight"])
    assert torch.equal(implicit["num_batches_tracked"], explicit["num_batches_tracked"])

    trimmed = aggregate_trimmed_mean(states, num_byzantine=1, device="cpu")
    assert torch.equal(trimmed["weight"], states[1]["weight"])


def test_weighted_fedavg_drops_nonfinite_rejected_updates() -> None:
    states = [_tiny_state(1.0), _tiny_state(float("nan")), _tiny_state(3.0)]

    # A zero-weight NaN must not poison the sum (0 * NaN is NaN in PyTorch).
    rejected = weighted_fedavg(states, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(rejected["weight"], states[0]["weight"])

    # If an invalid update is accidentally active, discard it and renormalize
    # the finite weights instead of returning a non-finite global model.
    recovered = weighted_fedavg(states, torch.tensor([0.25, 0.25, 0.5]))
    expected = (0.25 * states[0]["weight"] + 0.5 * states[2]["weight"]) / 0.75
    assert torch.allclose(recovered["weight"], expected)
    assert torch.isfinite(recovered["weight"]).all()


def test_multi_krum_scores_support_explicit_device() -> None:
    states = [_tiny_state(float(i)) for i in range(5)]
    implicit = compute_multi_krum_scores(states, num_byzantine=1)
    explicit = compute_multi_krum_scores(states, num_byzantine=1, device="cpu")
    assert torch.equal(implicit, explicit)


def test_upload_clipping_bounds_updates_and_neutralizes_nonfinite_uploads() -> None:
    reference = _tiny_state(0.0)
    states = [_tiny_state(6.0), _tiny_state(float("nan"))]

    clipped, stats = clip_client_updates(states, reference, max_norm=2.0)

    delta = clipped[0]["weight"] - reference["weight"]
    assert torch.linalg.vector_norm(delta).item() <= 2.0 + 1e-6
    assert torch.equal(clipped[1]["weight"], reference["weight"])
    assert stats["clipped_count"] == 2.0
    assert stats["nonfinite_replaced_count"] == 1.0


def test_base_defense_restores_last_finite_global_state() -> None:
    model_fn = lambda: nn.Linear(2, 1)
    defense = BaseDefense(FedConfig(), 1, torch.device("cpu"), model_fn)
    expected = defense.state_dict_for_clients()
    with torch.no_grad():
        defense.global_model.weight.fill_(float("nan"))

    restored = defense.state_dict_for_clients()

    assert torch.equal(restored["weight"], expected["weight"])
    assert torch.isfinite(restored["weight"]).all()
