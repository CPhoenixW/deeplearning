import torch

from src.attacks import ATTACK_REGISTRY, mixed_attack_for_client
from src.config import FedConfig, normalize_attack_name
from src.fixed_descriptor import FixedHierarchicalMultiViewDescriptor
from src.models import FashionCNN, LeNetClassifier
from src.tasks import TASK_REGISTRY
from src.defenses import DEFENSE_REGISTRY
from src.config import load_hyperparameter_table, resolve_hyperparameters


def test_mixed_attack_assignment_is_deterministic():
    cfg = FedConfig(num_clients=10, num_benign=6, mixed_attack_types="lf,bd,gn,lie")
    assert normalize_attack_name("hybrid") == "mix"
    assert [mixed_attack_for_client(cfg, cid) for cid in range(6, 10)] == ["lf", "bd", "gn", "lie"]
    assert "mix" in ATTACK_REGISTRY
    assert all(
        attack_id == "none" or attack.__module__.startswith("src.attacks.")
        for attack_id, attack in ATTACK_REGISTRY.items()
    )


def test_mnist_task_is_registered():
    assert "mnist" in TASK_REGISTRY


def test_grayscale_tasks_use_distinct_native_models_and_fixed_descriptor():
    expected_models = {
        "mnist": LeNetClassifier,
        "fashion_mnist": FashionCNN,
    }
    parameter_counts = {}
    for task_name, expected_model in expected_models.items():
        model = TASK_REGISTRY[task_name]().build_model()
        assert isinstance(model, expected_model)
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
        parameter_counts[task_name] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        assert parameter_counts[task_name] < 500_000
        assert not any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for module in model.modules()
        )

        reference = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        descriptor = FixedHierarchicalMultiViewDescriptor(
            reference,
            parameter_names=[name for name, _ in model.named_parameters()],
            output_dim=64,
            seed=7,
        )
        assert torch.equal(
            descriptor.describe(reference, reference),
            torch.zeros(64),
        )
    assert parameter_counts["fashion_mnist"] > parameter_counts["mnist"]


def test_modular_pipeline_registry_and_hyperparameters():
    assert "svdd" in DEFENSE_REGISTRY
    assert "dmc" in DEFENSE_REGISTRY
    assert all(
        defense.__module__.startswith("src.defenses.")
        for defense in DEFENSE_REGISTRY.values()
    )
    assert FedConfig().svdd_feature_mode == "fixed_projection"
    table = load_hyperparameter_table("configs/hyperparameters.json")
    values = resolve_hyperparameters(table, "lf", "svdd", "cifar10")
    assert values["svdd_feature_mode"] == "fixed_projection"
    assert values["svdd_loss_weight"] == 1.0
    assert "ag_news_svdd_features" not in values
