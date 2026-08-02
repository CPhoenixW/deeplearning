"""Smoke tests for the data-free multi-view FedDMC detector."""

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


def _states(server: DMCDefense, k: int = 6) -> list[dict[str, torch.Tensor]]:
    base = server.state_dict_for_clients()
    out = []
    for i in range(k):
        sd = copy.deepcopy(base)
        for name, value in sd.items():
            if value.is_floating_point():
                sd[name] = value + (0.01 * (i + 1)) * torch.randn_like(value)
        out.append(sd)
    return out


def test_dmc_registry_and_alias() -> None:
    assert "dmc" in DEFENSE_REGISTRY
    assert normalize_defense_name("FedDMC") == "dmc"
    assert normalize_defense_name("multi_view") == "dmc"


def test_dmc_shapes_and_normalized_weights() -> None:
    cfg = FedConfig(num_clients=6, num_benign=4, dmc_warmup_rounds=1)
    server = DMCDefense(cfg, d_bn=8, device=torch.device("cpu"), model_fn=_TinyNet)
    states = _states(server)
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), states)
    )
    assert stats.d.shape == (6,)
    assert stats.m.shape == (6,)
    assert stats.alpha.shape == (6,)
    assert torch.isfinite(stats.d).all()
    assert torch.isfinite(stats.alpha).all()
    assert abs(float(stats.alpha.sum()) - 1.0) < 1e-5
