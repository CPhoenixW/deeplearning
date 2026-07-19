"""Smoke tests for AlignIns and BNGuard defense servers."""

from __future__ import annotations

import copy

import torch

from src.config import FedConfig, normalize_defense_name
from src.server import DEFENSE_REGISTRY, AlignInsServer, BNGuardServer
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
    assert normalize_defense_name("align_ins") == "alignins"
    assert normalize_defense_name("bn_guard") == "bnguard"


def test_alignins_aggregate_smoke() -> None:
    cfg = FedConfig(total_rounds=1, num_clients=5, num_benign=4)
    task = TASK_REGISTRY["cifar10"]()
    device = torch.device("cpu")

    def model_fn():
        return task.build_model()

    server = AlignInsServer(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(round_idx=1, client_state_dicts=client_sds)

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

    server = BNGuardServer(cfg, d_bn=128, device=device, model_fn=model_fn)
    client_sds = _make_client_states(server.global_model, k=5)
    stats = server.aggregate(round_idx=1, client_state_dicts=client_sds)

    assert stats.d.shape == (5,)
    assert stats.m.shape == (5,)
    assert stats.alpha.shape == (5,)
    assert float(stats.m.sum().item()) >= 1.0
    assert abs(float(stats.alpha.sum().item()) - 1.0) < 1e-5
    assert torch.isfinite(stats.d).all()


if __name__ == "__main__":
    test_registry_and_aliases()
    test_alignins_aggregate_smoke()
    test_bnguard_aggregate_smoke()
    print("OK: defense smoke tests passed")
