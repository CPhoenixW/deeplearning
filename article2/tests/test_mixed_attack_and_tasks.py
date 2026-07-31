from src.clients import ATTACK_REGISTRY, mixed_attack_for_client
from src.config import FedConfig, normalize_attack_name
from src.tasks import TASK_REGISTRY


def test_mixed_attack_assignment_is_deterministic():
    cfg = FedConfig(num_clients=10, num_benign=6, mixed_attack_types="lf,bd,gn,lie")
    assert normalize_attack_name("hybrid") == "mix"
    assert [mixed_attack_for_client(cfg, cid) for cid in range(6, 10)] == ["lf", "bd", "gn", "lie"]
    assert "mix" in ATTACK_REGISTRY


def test_mnist_task_is_registered():
    assert "mnist" in TASK_REGISTRY

