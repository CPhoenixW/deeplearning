"""Tests for the unified 4096-dim direct-parameter SVDD inputs."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.config import FedConfig
from src.defenses import SVDDDefense


class _LargeEnoughModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.feature(x)))


def _server(mode: str) -> SVDDDefense:
    config = FedConfig(
        num_clients=5,
        num_benign=4,
        latent_dim=8,
        svdd_input_mode=mode,
        svdd_input_dim=4096,
        device="cpu",
    )
    return SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_LargeEnoughModel,
    )


def _clients(reference: dict[str, torch.Tensor], k: int = 5) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(314159)
    states: list[dict[str, torch.Tensor]] = []
    for idx in range(k):
        state = copy.deepcopy(reference)
        for name, value in state.items():
            if value.is_floating_point():
                state[name] = value + (0.01 + 0.005 * idx) * torch.randn(
                    value.shape, generator=generator, dtype=value.dtype
                )
        states.append(state)
    return states


def test_absolute_and_delta_share_the_same_4096_coordinates() -> None:
    absolute = _server("absolute")
    delta = _server("delta")
    delta.global_model.load_state_dict(absolute.global_model.state_dict())
    reference = absolute.state_dict_for_clients()
    clients = _clients(reference)

    x_absolute = absolute._build_input_matrix(clients)
    x_delta = delta._build_input_matrix(clients)
    full_reference = torch.cat(
        [reference[name].float().reshape(-1) for name in absolute.param_names]
    )
    expected_delta = x_absolute - full_reference.index_select(0, absolute._parameter_indices)
    assert x_absolute.shape == (5, 4096)
    assert torch.allclose(x_delta, expected_delta, atol=1e-6)


def test_mean_std_normalization_is_shift_invariant_between_modes() -> None:
    absolute = _server("absolute")
    delta = _server("delta")
    delta.global_model.load_state_dict(absolute.global_model.state_dict())
    reference = absolute.state_dict_for_clients()
    clients = _clients(reference)

    x_absolute, finite_absolute = absolute._scale_input_matrix(
        absolute._build_input_matrix(clients)
    )
    x_delta, finite_delta = delta._scale_input_matrix(delta._build_input_matrix(clients))
    assert torch.equal(finite_absolute, finite_delta)
    assert torch.allclose(x_absolute, x_delta, atol=1e-5, rtol=1e-5)


def test_nonfinite_client_is_excluded_from_mean_std() -> None:
    server = _server("absolute")
    reference = server.state_dict_for_clients()
    clients = _clients(reference)
    clients[0][server.param_names[0]] = clients[0][server.param_names[0]].clone()
    clients[0][server.param_names[0]].reshape(-1)[0] = float("nan")

    raw = server._build_input_matrix(clients)
    normalized, finite = server._scale_input_matrix(raw)
    assert not bool(finite[0].item())
    assert torch.equal(normalized[0], torch.zeros(4096))
    assert bool(torch.isfinite(normalized).all().item())


def test_models_with_fewer_than_4096_parameters_fail_fast() -> None:
    with pytest.raises(ValueError, match="at least 4096"):
        SVDDDefense(
            FedConfig(device="cpu"),
            d_bn=4096,
            device=torch.device("cpu"),
            model_fn=lambda: nn.Linear(4, 2),
        )
