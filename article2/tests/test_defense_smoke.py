"""Smoke tests for the canonical defense strategies."""

from __future__ import annotations

import copy

import torch

from src.config import FedConfig, normalize_defense_name
from src.defenses import (
    DEFENSE_REGISTRY,
    AlignInsDefense,
    BNGuardDefense,
    DefenseContext,
    FLANDERSDefense,
    FLGMMDefense,
)
from src.tasks import TASK_REGISTRY


def _make_client_states(model: torch.nn.Module, k: int = 5) -> list[dict[str, torch.Tensor]]:
    base = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    states = []
    for i in range(k):
        sd = copy.deepcopy(base)
        for name, tensor in sd.items():
            if tensor.is_floating_point():
                sd[name] = tensor + 0.01 * float(i + 1) * torch.randn_like(tensor)
        states.append(sd)
    return states


def test_registry_and_aliases() -> None:
    assert "alignins" in DEFENSE_REGISTRY
    assert "bnguard" in DEFENSE_REGISTRY
    assert "flgmm" in DEFENSE_REGISTRY
    assert "flanders" in DEFENSE_REGISTRY
    assert normalize_defense_name("align_ins") == "alignins"
    assert normalize_defense_name("bn_guard") == "bnguard"
    assert normalize_defense_name("fl_gmm") == "flgmm"


def test_alignins_aggregate_smoke() -> None:
    cfg = FedConfig(total_rounds=1, num_clients=5, num_benign=4)
    task = TASK_REGISTRY["cifar10"]()
    device = torch.device("cpu")

    def model_fn():
        return task.build_model()

    server = AlignInsDefense(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), client_sds)
    )

    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert stats.alpha.shape == (5,)
    assert float(stats.m.sum().item()) >= 1.0
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()


def test_bnguard_aggregate_smoke() -> None:
    cfg = FedConfig(total_rounds=1, num_clients=5, num_benign=4)
    task = TASK_REGISTRY["cifar10"]()
    device = torch.device("cpu")

    def model_fn():
        return task.build_model()

    server = BNGuardDefense(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), client_sds)
    )

    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert stats.alpha.shape == (5,)
    assert float(stats.m.sum().item()) >= 1.0
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()


def test_flgmm_aggregate_smoke() -> None:
    cfg = FedConfig(total_rounds=2, num_clients=5, num_benign=4)
    cfg.flgmm_warmup_rounds = 1
    cfg.flgmm_em_iters = 2
    task = TASK_REGISTRY["cifar10"]()
    device = torch.device("cpu")

    def model_fn():
        return task.build_model()

    server = FLGMMDefense(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), client_sds)
    )
    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()

    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(2, server.state_dict_for_clients(), client_sds)
    )
    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()


def test_flanders_aggregate_smoke() -> None:
    cfg = FedConfig(total_rounds=2, num_clients=5, num_benign=4)
    cfg.flanders_sampling = 8
    cfg.flanders_window = 2
    cfg.flanders_maxiter = 1
    task = TASK_REGISTRY["cifar10"]()
    device = torch.device("cpu")

    def model_fn():
        return task.build_model()

    server = FLANDERSDefense(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(1, server.state_dict_for_clients(), client_sds)
    )
    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()

    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(
        DefenseContext(2, server.state_dict_for_clients(), client_sds)
    )
    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()


if __name__ == "__main__":
    test_registry_and_aliases()
    test_alignins_aggregate_smoke()
    test_bnguard_aggregate_smoke()
    test_flgmm_aggregate_smoke()
    test_flanders_aggregate_smoke()
    print("OK: defense smoke tests passed")
