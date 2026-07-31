"""Tests for absolute-model versus model-delta AE-SVDD inputs."""

from __future__ import annotations

import copy

import torch
from torch import nn

from src.config import FedConfig
from src.server import SVDDServer
from src.utils import (
    build_svdd_feature_matrix,
    extract_bn_features,
    robust_scale_features,
)


class _TinyBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn1(self.fc1(x))))


def _make_distinct_client_states(
    reference: dict[str, torch.Tensor], k: int = 5
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(314159)
    states: list[dict[str, torch.Tensor]] = []
    for idx in range(k):
        sd = copy.deepcopy(reference)
        for name, value in sd.items():
            if value.is_floating_point():
                noise = torch.randn(
                    value.shape,
                    generator=generator,
                    dtype=value.dtype,
                )
                sd[name] = value + (0.01 + 0.005 * idx) * noise
        states.append(sd)
    return states


def test_absolute_and_delta_feature_relationship() -> None:
    model = _TinyBNModel()
    reference = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    clients = _make_distinct_client_states(reference)

    x_absolute = build_svdd_feature_matrix(
        clients,
        extract_bn_features,
        input_mode="absolute",
    )
    x_delta = build_svdd_feature_matrix(
        clients,
        extract_bn_features,
        input_mode="delta",
        reference_state_dict=reference,
    )
    reference_features = extract_bn_features(reference)

    assert torch.allclose(
        x_absolute - reference_features.unsqueeze(0),
        x_delta,
        atol=1e-6,
        rtol=1e-5,
    )


def test_per_round_median_mad_scaling_makes_modes_equivalent() -> None:
    model = _TinyBNModel()
    reference = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    clients = _make_distinct_client_states(reference)

    x_absolute = build_svdd_feature_matrix(
        clients,
        extract_bn_features,
        input_mode="absolute",
    )
    x_delta = build_svdd_feature_matrix(
        clients,
        extract_bn_features,
        input_mode="delta",
        reference_state_dict=reference,
    )

    assert torch.allclose(
        robust_scale_features(x_absolute),
        robust_scale_features(x_delta),
        atol=2e-4,
        rtol=2e-4,
    )


def test_svdd_phase1_outputs_match_between_input_modes() -> None:
    torch.manual_seed(7)
    cfg_absolute = FedConfig(
        num_clients=5,
        num_benign=4,
        total_rounds=1,
        phase1_rounds=1,
        ae_warmup_keep_ratio=1.0,
        latent_dim=8,
        svdd_input_mode="absolute",
        device="cpu",
    )
    cfg_delta = copy.deepcopy(cfg_absolute)
    cfg_delta.svdd_input_mode = "delta"

    def model_fn() -> nn.Module:
        return _TinyBNModel()

    d_bn = extract_bn_features(model_fn().state_dict()).numel()
    server_absolute = SVDDServer(
        cfg_absolute,
        d_bn=d_bn,
        device=torch.device("cpu"),
        model_fn=model_fn,
        svdd_feature_extractor=extract_bn_features,
    )
    server_delta = SVDDServer(
        cfg_delta,
        d_bn=d_bn,
        device=torch.device("cpu"),
        model_fn=model_fn,
        svdd_feature_extractor=extract_bn_features,
    )
    server_delta.global_model.load_state_dict(server_absolute.global_model.state_dict())
    server_delta.ae.load_state_dict(server_absolute.ae.state_dict())

    reference = server_absolute.state_dict_for_clients()
    clients = _make_distinct_client_states(reference)

    stats_absolute = server_absolute.aggregate(1, copy.deepcopy(clients))
    stats_delta = server_delta.aggregate(1, copy.deepcopy(clients))

    assert torch.allclose(stats_absolute.d, stats_delta.d, atol=2e-4, rtol=2e-4)
    assert torch.equal(stats_absolute.m, stats_delta.m)
    assert torch.allclose(stats_absolute.alpha, stats_delta.alpha)
    assert abs(stats_absolute.ae_loss - stats_delta.ae_loss) < 2e-4

    for key, absolute_value in server_absolute.state_dict_for_clients().items():
        delta_value = server_delta.state_dict_for_clients()[key]
        if absolute_value.is_floating_point():
            assert torch.allclose(
                absolute_value,
                delta_value,
                atol=1e-6,
                rtol=1e-5,
            )
        else:
            assert torch.equal(absolute_value, delta_value)

