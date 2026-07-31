from __future__ import annotations

import copy

import torch
from torch import nn

from src.config import FedConfig
from src.fixed_descriptor import FixedHierarchicalMultiViewDescriptor
from src.server import SVDDServer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn1(self.fc1(x))))


def _reference() -> tuple[_TinyModel, dict[str, torch.Tensor]]:
    model = _TinyModel()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return model, state


def test_descriptor_is_fixed_and_has_exact_dimension() -> None:
    model, reference = _reference()
    names = [name for name, _ in model.named_parameters()]
    client = copy.deepcopy(reference)
    for name in names:
        client[name] = client[name] + 0.01 * torch.arange(
            client[name].numel(), dtype=client[name].dtype
        ).reshape(client[name].shape)

    first = FixedHierarchicalMultiViewDescriptor(
        reference,
        parameter_names=names,
        output_dim=64,
        seed=17,
    )
    second = FixedHierarchicalMultiViewDescriptor(
        reference,
        parameter_names=names,
        output_dim=64,
        seed=17,
    )
    x_first = first.describe(client, reference)
    x_second = second.describe(client, reference)

    assert x_first.shape == (64,)
    assert torch.equal(x_first, x_second)
    assert torch.count_nonzero(x_first) > 0
    assert first.layout.parameter_count == sum(p.numel() for p in model.parameters())


def test_zero_update_maps_to_zero() -> None:
    model, reference = _reference()
    descriptor = FixedHierarchicalMultiViewDescriptor(
        reference,
        parameter_names=[name for name, _ in model.named_parameters()],
        output_dim=64,
    )
    assert torch.equal(descriptor.describe(reference, reference), torch.zeros(64))


def test_fixed_descriptor_feeds_svdd_phase1() -> None:
    cfg = FedConfig(
        num_clients=3,
        num_benign=2,
        phase1_rounds=1,
        latent_dim=8,
        svdd_feature_mode="fixed_projection",
        param_descriptor_dim=64,
        param_descriptor_seed=31,
        param_descriptor_device="cpu",
        device="cpu",
    )

    def model_fn() -> nn.Module:
        return _TinyModel()

    server = SVDDServer(
        cfg,
        d_bn=64,
        device=torch.device("cpu"),
        model_fn=model_fn,
    )
    reference = server.state_dict_for_clients()
    clients = []
    for idx in range(cfg.num_clients):
        state = copy.deepcopy(reference)
        for name, value in state.items():
            if value.is_floating_point():
                state[name] = value + (idx + 1) * 0.001
        clients.append(state)

    stats = server.aggregate(1, clients)
    assert stats.d.shape == (cfg.num_clients,)
    assert stats.m.shape == (cfg.num_clients,)
    assert torch.isfinite(stats.d).all()
