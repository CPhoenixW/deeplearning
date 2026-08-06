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
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_both_phases_use_validation_topk_and_alpha() -> None:
    torch.manual_seed(7)
    validation = DataLoader(
        TensorDataset(torch.randn(64, 4), torch.randint(0, 2, (64,))),
        batch_size=16,
    )
    config = FedConfig(
        num_clients=5,
        num_benign=4,
        phase1_rounds=1,
        latent_dim=4,
        param_descriptor_dim=64,
        param_descriptor_device="cpu",
        svdd_feature_mode="fixed_projection",
        device="cpu",
        alpha=0.25,
    )
    server = SVDDDefense(
        config,
        d_bn=64,
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
