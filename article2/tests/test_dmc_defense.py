"""Unit tests for the FedDMC (TDSC 2024) reproduction."""

from __future__ import annotations

import copy

import torch
from torch import nn

from src.config import FedConfig, normalize_defense_name
from src.defenses import DEFENSE_REGISTRY, DMCDefense, DefenseContext


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _shifted_states(
    server: DMCDefense,
    shifts: list[float],
) -> list[dict[str, torch.Tensor]]:
    base = server.state_dict_for_clients()
    states = []
    for shift in shifts:
        state = copy.deepcopy(base)
        for name, value in state.items():
            if value.is_floating_point():
                state[name] = value + float(shift)
        states.append(state)
    return states


def test_dmc_registry_and_alias() -> None:
    assert "dmc" in DEFENSE_REGISTRY
    assert normalize_defense_name("FedDMC") == "dmc"


def test_dmc_shapes_and_uniform_kept_weights() -> None:
    cfg = FedConfig(num_clients=10, num_benign=7, dmc_ema_decay=0.8)
    server = DMCDefense(
        cfg,
        d_bn=8,
        device=torch.device("cpu"),
        model_fn=_TinyNet,
    )
    # Seven close benign uploads and three distant malicious uploads.  The
    # malicious group is exactly the paper default min_cluster_size=3.
    states = _shifted_states(
        server,
        [0.000, 0.002, -0.002, 0.004, -0.004, 0.006, -0.006, 4.0, 4.2, 3.8],
    )
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), states)
    )

    assert stats.d.shape == (10,)
    assert stats.m.shape == (10,)
    assert stats.alpha.shape == (10,)
    assert torch.isfinite(stats.d).all()
    assert torch.isfinite(stats.alpha).all()
    assert abs(float(stats.alpha.sum()) - 1.0) < 1e-5
    assert int(stats.m[:7].sum().item()) == 7
    assert int(stats.m[7:].sum().item()) == 0
    kept_weights = stats.alpha[stats.m >= 0.5]
    assert torch.allclose(
        kept_weights,
        torch.full_like(kept_weights, 1.0 / 7.0),
        atol=1e-6,
    )


def test_sedc_uses_history_instead_of_single_round_only() -> None:
    cfg = FedConfig(num_clients=10, num_benign=7, dmc_ema_decay=0.8)
    server = DMCDefense(
        cfg,
        d_bn=8,
        device=torch.device("cpu"),
        model_fn=_TinyNet,
    )
    first = _shifted_states(
        server,
        [0.000, 0.002, -0.002, 0.004, -0.004, 0.006, -0.006, 4.0, 4.2, 3.8],
    )
    first_stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), first)
    )
    first_trust = first_stats.participant_metrics["feddmc_trust"].clone()

    # Feed a round with no separable client variation.  BTBCN returns all
    # clients benign, but the previous malicious clients should retain lower
    # trust because SEDC is an EMA over historical detections.
    second = _shifted_states(server, [0.0] * 10)
    second_stats = server.aggregate(
        DefenseContext(2, server.state_dict_for_clients(), second)
    )
    second_trust = second_stats.participant_metrics["feddmc_trust"]

    assert torch.all(second_trust[7:] > first_trust[7:])
    assert torch.all(second_trust[7:] < second_trust[:7].mean())
