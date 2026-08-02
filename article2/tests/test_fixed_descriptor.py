from __future__ import annotations

import copy

import torch
from torch import nn

from src.config import FedConfig
from src.defenses import DefenseContext, SVDDDefense
from src.defenses.svdd import _lower_quantile_mask
from src.fixed_descriptor import FixedHierarchicalMultiViewDescriptor


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
    assert first.layout.global_dim == 32
    assert first.layout.layer_dim == 24
    assert first.layout.statistics_dim == 8
    assert first.layout.parameter_count == sum(p.numel() for p in model.parameters())


def test_zero_update_maps_to_zero() -> None:
    model, reference = _reference()
    descriptor = FixedHierarchicalMultiViewDescriptor(
        reference,
        parameter_names=[name for name, _ in model.named_parameters()],
        output_dim=64,
    )
    assert torch.equal(descriptor.describe(reference, reference), torch.zeros(64))


def test_descriptor_view_ratios_are_configurable() -> None:
    model, reference = _reference()
    names = [name for name, _ in model.named_parameters()]
    client = copy.deepcopy(reference)
    for name in names:
        client[name] = client[name] + 0.01

    descriptor = FixedHierarchicalMultiViewDescriptor(
        reference,
        parameter_names=names,
        output_dim=64,
        global_ratio=0.25,
        layer_ratio=0.5,
        statistics_ratio=0.25,
    )
    assert descriptor.layout.global_dim == 16
    assert descriptor.layout.layer_dim == 32
    assert descriptor.layout.statistics_dim == 16
    assert descriptor.describe(client, reference).shape == (64,)

    for ratios, expected_dims in (
        ((1.0, 0.0, 0.0), (64, 0, 0)),
        ((0.0, 1.0, 0.0), (0, 64, 0)),
        ((0.0, 0.0, 1.0), (0, 0, 64)),
    ):
        single_view = FixedHierarchicalMultiViewDescriptor(
            reference,
            parameter_names=names,
            output_dim=64,
            global_ratio=ratios[0],
            layer_ratio=ratios[1],
            statistics_ratio=ratios[2],
        )
        assert (
            single_view.layout.global_dim,
            single_view.layout.layer_dim,
            single_view.layout.statistics_dim,
        ) == expected_dims
        assert single_view.describe(client, reference).shape == (64,)

    try:
        FixedHierarchicalMultiViewDescriptor(
            reference,
            parameter_names=names,
            output_dim=64,
            global_ratio=0.5,
            layer_ratio=0.5,
            statistics_ratio=0.5,
        )
    except ValueError as error:
        assert "sum to 1.0" in str(error)
    else:
        raise AssertionError("Invalid descriptor view ratios were accepted.")

    many_tensor_model = nn.Module()
    many_tensor_model.weights = nn.ParameterList(
        [nn.Parameter(torch.randn(2)) for _ in range(40)]
    )
    many_reference = {
        key: value.detach().cpu().clone()
        for key, value in many_tensor_model.state_dict().items()
    }
    many_client = {
        key: value + 0.01 for key, value in many_reference.items()
    }
    low_dimensional = FixedHierarchicalMultiViewDescriptor(
        many_reference,
        parameter_names=[name for name, _ in many_tensor_model.named_parameters()],
        output_dim=64,
    )
    assert low_dimensional.layout.layer_dim == 24
    assert low_dimensional.layout.layer_count == 40
    assert low_dimensional.describe(many_client, many_reference).shape == (64,)


def test_trusted_sample_quantiles_are_configurable_and_validated() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert torch.equal(
        _lower_quantile_mask(values, 0.5, parameter_name="center_init_quantile"),
        torch.tensor([True, True, True, False, False]),
    )
    assert torch.equal(
        _lower_quantile_mask(values, 0.8, parameter_name="phase2_recon_quantile"),
        torch.tensor([True, True, True, True, False]),
    )
    for invalid in (0.0, -0.1, 1.1, float("nan")):
        try:
            _lower_quantile_mask(values, invalid, parameter_name="test_quantile")
        except ValueError as error:
            assert "must be in (0, 1]" in str(error)
        else:
            raise AssertionError(f"Invalid quantile {invalid!r} was accepted.")


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
        param_descriptor_global_ratio=0.25,
        param_descriptor_layer_ratio=0.5,
        param_descriptor_statistics_ratio=0.25,
        center_init_quantile=0.3,
        phase2_recon_quantile=0.6,
        device="cpu",
    )

    def model_fn() -> nn.Module:
        return _TinyModel()

    server = SVDDDefense(
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

    stats = server.aggregate(
        DefenseContext(1, reference, clients)
    )
    assert stats.d.shape == (cfg.num_clients,)
    assert stats.m.shape == (cfg.num_clients,)
    assert torch.isfinite(stats.d).all()
    assert server._fixed_descriptor is not None
    assert server._fixed_descriptor.layout.global_dim == 16
    assert server._fixed_descriptor.layout.layer_dim == 32
    assert server._fixed_descriptor.layout.statistics_dim == 16

    phase2_reference = server.state_dict_for_clients()
    phase2_clients = []
    for idx in range(cfg.num_clients):
        state = copy.deepcopy(phase2_reference)
        for name, value in state.items():
            if value.is_floating_point():
                state[name] = value + (idx + 1) * 0.0005
        phase2_clients.append(state)
    phase2_stats = server.aggregate(
        DefenseContext(2, phase2_reference, phase2_clients)
    )
    assert phase2_stats.d.shape == (cfg.num_clients,)
    assert torch.isfinite(phase2_stats.d).all()
