from __future__ import annotations

import torch
from torch import nn

from src.clients import GaussianNoiseClient, LieAttackClient
from src.config import FedConfig, MatrixRunConfig
from src.utils import aggregate_trimmed_mean, compute_multi_krum_scores, weighted_fedavg


def _tiny_state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value, value + 1.0]),
        "num_batches_tracked": torch.tensor(3, dtype=torch.long),
    }


def _unused_model() -> nn.Module:
    raise AssertionError("attack must not build or train a local model")


def test_matrix_defaults_enable_fast_path() -> None:
    cfg = MatrixRunConfig()
    # On RTX 4060 + 32x32 CIFAR batches, AMP/channels-last are slower; keep
    # them available as opt-ins for larger workloads.
    assert not cfg.use_amp
    assert not cfg.channels_last
    assert cfg.cuda_aggregation
    assert cfg.reuse_client_model
    assert cfg.skip_redundant_attack_training
    assert not cfg.round_diagnostics


def test_gaussian_and_lie_skip_discarded_local_training() -> None:
    cfg = FedConfig(
        num_clients=2,
        num_benign=1,
        gaussian_sigma=0.0,
        skip_redundant_attack_training=True,
    )
    reference = _tiny_state(2.0)
    loader = []

    gaussian = GaussianNoiseClient(1, torch.device("cpu"), cfg, loader, _unused_model)
    gaussian_upload = gaussian.local_step(reference)
    assert gaussian_upload.keys() == reference.keys()

    lie = LieAttackClient(1, torch.device("cpu"), cfg, loader, _unused_model)
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


def test_multi_krum_scores_support_explicit_device() -> None:
    states = [_tiny_state(float(i)) for i in range(5)]
    implicit = compute_multi_krum_scores(states, num_byzantine=1)
    explicit = compute_multi_krum_scores(states, num_byzantine=1, device="cpu")
    assert torch.equal(implicit, explicit)
