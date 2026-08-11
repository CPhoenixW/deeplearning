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
    assert normalize_attack_name("min-max") == "minmax"
    assert normalize_attack_name("min_sum") == "minsum"
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
    assert values["alpha"] == 0.5
    assert "ag_news_svdd_features" not in values
    assert values["param_descriptor_global_ratio"] == 0.5
    assert values["param_descriptor_layer_ratio"] == 0.375
    assert values["param_descriptor_statistics_ratio"] == 0.125
    assert values["center_init_quantile"] == 0.5
    assert values["phase2_recon_quantile"] == 0.8


def test_stage_a_selected_client_hyperparameters_are_task_specific():
    table = load_hyperparameter_table("configs/hyperparameters.json")
    expected = {
        "mnist": (0.1, 0.0001),
        "fashion_mnist": (0.1, 0.0),
        "cifar10": (0.05, 0.0001),
        "ag_news": (0.1, 0.0),
    }
    for task, (client_lr, client_weight_decay) in expected.items():
        values = resolve_hyperparameters(table, "none", "avg", task)
        assert values["client_lr"] == client_lr
        assert values["client_weight_decay"] == client_weight_decay
