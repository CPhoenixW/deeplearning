"""Focused tests for the two-phase validation-driven Top-K selector."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import FedConfig
from src.defenses import DefenseContext, SVDDDefense


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.feature(inputs)))


def test_both_phases_use_validation_topk_and_svdd_lambda() -> None:
    torch.manual_seed(7)
    validation = DataLoader(
        TensorDataset(torch.randn(64, 64), torch.randint(0, 2, (64,))),
        batch_size=16,
    )
    config = FedConfig(
        num_clients=5,
        num_benign=4,
        phase1_rounds=1,
        latent_dim=4,
        device="cpu",
        svdd_lambda=0.25,
    )
    server = SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_TinyModel,
        validation_loader=validation,
    )
    reference = server.state_dict_for_clients()
    clients = []
    for index in range(config.num_clients):
        state = copy.deepcopy(reference)
        for name, value in state.items():
            if value.is_floating_point():
                state[name] = value + 0.01 * (index + 1)
        clients.append(state)

    phase1 = server.aggregate(DefenseContext(1, reference, clients))
    phase2 = server.aggregate(
        DefenseContext(2, server.state_dict_for_clients(), clients)
    )

    for result in (phase1, phase2):
        assert result.phase in {"warmup", "filtering"}
        assert len(result.server_metrics["validation_candidates"]) == 5
        assert result.server_metrics["selected_reject_ratio"] in {
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        }
        assert int(result.m.sum().item()) in {2, 3, 4}
    expected = phase2.svdd_loss * 0.25 + phase2.recon_loss * 0.75
    assert abs(phase2.total_loss - expected) < 1e-6


def test_validation_ties_choose_largest_rejection_ratio() -> None:
    """A flat validation score must favor the more defensive candidate."""

    config = FedConfig(
        num_clients=5,
        num_benign=4,
        latent_dim=4,
        device="cpu",
    )
    validation = DataLoader(
        TensorDataset(torch.zeros(10, 64), torch.zeros(10, dtype=torch.long)),
        batch_size=10,
    )
    server = SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_TinyModel,
        validation_loader=validation,
    )
    reference = server.state_dict_for_clients()
    clients = [copy.deepcopy(reference) for _ in range(config.num_clients)]
    scores = torch.arange(config.num_clients, dtype=torch.float32)

    _, _, selected_ratio, _, candidates = server._select_topk_by_validation(
        scores, clients
    )

    assert set(candidates.values()) == {candidates["0.10"]}
    assert selected_ratio == 0.5


def test_minimum_rejection_can_win_validation_selection() -> None:
    config = FedConfig(
        num_clients=5,
        num_benign=5,
        latent_dim=4,
        device="cpu",
    )
    validation = DataLoader(
        TensorDataset(torch.zeros(10, 64), torch.zeros(10, dtype=torch.long)),
        batch_size=10,
    )
    server = SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_TinyModel,
        validation_loader=validation,
    )
    reference = server.state_dict_for_clients()
    clients = [copy.deepcopy(reference) for _ in range(config.num_clients)]
    calls = []

    def prefer_first_candidate() -> float:
        calls.append(len(calls))
        return 1.0 - 0.01 * len(calls)

    server._validation_accuracy = prefer_first_candidate  # type: ignore[method-assign]
    _, _, selected_ratio, _, candidates = server._select_topk_by_validation(
        torch.arange(config.num_clients, dtype=torch.float32), clients
    )

    assert len(calls) == 5
    assert set(candidates) == {"0.10", "0.20", "0.30", "0.40", "0.50"}
    assert selected_ratio == 0.1


def test_phase_scores_are_independent_from_svdd_lambda() -> None:
    config = FedConfig(
        phase1_score_mode="recon",
        phase2_score_mode="combined",
        svdd_lambda=0.5,
        num_clients=5,
        num_benign=5,
        latent_dim=4,
        device="cpu",
    )
    server = SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_TinyModel,
    )
    assert server.phase1_score_mode == "recon"
    assert server.phase2_score_mode == "combined"
    assert config.svdd_lambda == 0.5


def test_configured_latent_dimension_is_preserved() -> None:
    config = FedConfig(
        latent_dim=512,
        device="cpu",
    )
    server = SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_TinyModel,
    )
    assert server.ae.encoder.net[-1].out_features == 512
