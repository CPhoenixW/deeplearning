"""Tests for the 4096-D fixed-descriptor SVDD inputs."""

from __future__ import annotations

import copy

import torch
from torch import nn

from src.config import FedConfig
from src.defenses import SVDDDefense


class _SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(4, 3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.feature(x)))


def _server(mode: str) -> SVDDDefense:
    config = FedConfig(
        num_clients=5,
        num_benign=4,
        latent_dim=8,
        svdd_input_mode=mode,
        svdd_input_dim=4096,
        svdd_descriptor_device="cpu",
        device="cpu",
    )
    return SVDDDefense(
        config,
        d_bn=4096,
        device=torch.device("cpu"),
        model_fn=_SmallModel,
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


def test_both_modes_return_a_4096_dimensional_descriptor() -> None:
    absolute = _server("absolute")
    delta = _server("delta")
    delta.global_model.load_state_dict(absolute.global_model.state_dict())
    reference = absolute.state_dict_for_clients()
    clients = _clients(reference)

    x_absolute = absolute._build_input_matrix(clients)
    x_delta = delta._build_input_matrix(clients)
    expected_absolute = absolute.descriptor.describe_many(clients, absolute._zero_reference)
    expected_delta = delta.descriptor.describe_many(clients, reference)

    assert x_absolute.shape == (5, 4096)
    assert x_delta.shape == (5, 4096)
    assert torch.allclose(x_absolute, expected_absolute)
    assert torch.allclose(x_delta, expected_delta)
    assert not torch.allclose(x_absolute, x_delta)


def test_descriptor_is_deterministic() -> None:
    first = _server("absolute")
    second = _server("absolute")
    reference = first.state_dict_for_clients()
    clients = _clients(reference)
    second.global_model.load_state_dict(reference)

    assert torch.equal(
        first._build_input_matrix(clients),
        second._build_input_matrix(clients),
    )


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


def test_models_smaller_than_4096_parameters_are_supported() -> None:
    server = _server("absolute")
    assert sum(parameter.numel() for parameter in server.global_model.parameters()) < 4096
    assert server.descriptor.layout.output_dim == 4096
