from src.clients import ATTACK_REGISTRY, mixed_attack_for_client
from src.config import FedConfig, normalize_attack_name
from src.tasks import TASK_REGISTRY
from src.defenses import DEFENSE_REGISTRY
from src.config import load_hyperparameter_table, resolve_hyperparameters


def test_mixed_attack_assignment_is_deterministic():
    cfg = FedConfig(num_clients=10, num_benign=6, mixed_attack_types="lf,bd,gn,lie")
    assert normalize_attack_name("hybrid") == "mix"
    assert [mixed_attack_for_client(cfg, cid) for cid in range(6, 10)] == ["lf", "bd", "gn", "lie"]
    assert "mix" in ATTACK_REGISTRY


def test_mnist_task_is_registered():
    assert "mnist" in TASK_REGISTRY


def test_modular_pipeline_registry_and_hyperparameters():
    assert "svdd" in DEFENSE_REGISTRY
    assert "dmc" in DEFENSE_REGISTRY
    table = load_hyperparameter_table("configs/hyperparameters.json")
    values = resolve_hyperparameters(table, "lf", "svdd", "cifar10")
    assert values["svdd_feature_mode"] == "fixed_projection"
    assert values["svdd_loss_weight"] == 1.0
