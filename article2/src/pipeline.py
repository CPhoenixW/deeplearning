from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Sequence

from .attacks import ATTACK_REGISTRY, validate_attack_config
from .config import (
    FedConfig,
    apply_fed_config_overrides,
    load_fed_config_values,
    load_hyperparameter_table,
    load_pipeline_config,
    normalize_attack_name,
    normalize_defense_name,
    resolve_fed_config_path,
    resolve_hyperparameters,
    resolve_hyperparameters_path,
)
from .defenses import DEFENSE_REGISTRY
from .pipeline_core.contracts import PipelineContext
from .pipeline_core.result import StructuredResultWriter
from .pipeline_core.runner import run_pipeline
from .tasks import TASK_REGISTRY, get_task


def _split(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate(values: Sequence[str], registry: Dict[str, object], label: str) -> None:
    for value in values:
        if value not in registry:
            raise ValueError(f"Unknown {label} {value!r}; available: {sorted(registry)}")


def _apply_defense(cfg: FedConfig, defense: str) -> None:
    cfg.defense_type = defense
    if defense != "svdd":
        cfg.aggregation_method = defense


def _run_combo(
    base_cfg: FedConfig,
    task_name: str,
    attack: str,
    defense: str,
    output_dir: Path,
    prepared_dataloaders,
    table,
    config_files: Dict[str, str],
    overrides: dict[str, object],
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.task_name = task_name
    cfg.attack_type = attack
    _apply_defense(cfg, defense)
    applied = resolve_hyperparameters(table, attack, defense, task_name)
    apply_fed_config_overrides(cfg, applied, source=f"hyperparameters.{attack}.{defense}")
    apply_fed_config_overrides(cfg, overrides, source="pipeline overrides")
    if attack == "none":
        # A clean control has no synthetic malicious identities. This keeps the
        # offline TPR/FPR labels consistent with the actual client behavior.
        cfg.num_benign = cfg.num_clients
    validate_attack_config(attack, cfg)
    context = PipelineContext(
        config=cfg,
        task_name=task_name,
        attack_name=attack,
        defense_name=defense,
        output_dir=output_dir,
        config_files=config_files,
        applied_hyperparameters=applied,
        prepared_dataloaders=prepared_dataloaders,
    )
    writer = StructuredResultWriter(context)
    context.round_observer = writer.observe
    run_pipeline(context)
    return writer.write()


def main() -> None:
    parser = argparse.ArgumentParser(description="Composable federated experiment pipeline")
    parser.add_argument("--config", required=True, help="Pipeline configuration JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print combinations without loading data or training.")
    args = parser.parse_args()
    matrix = load_pipeline_config(args.config)
    fed_path = resolve_fed_config_path(matrix.fed_config_file)
    hyper_path = resolve_hyperparameters_path(matrix.hyperparameters_file)
    base = FedConfig()
    apply_fed_config_overrides(base, load_fed_config_values(fed_path), source=str(fed_path))
    apply_fed_config_overrides(
        base,
        matrix.fed_config_overrides,
        source="pipeline overrides",
    )
    table = load_hyperparameter_table(hyper_path)
    tasks = sorted(TASK_REGISTRY) if matrix.task.strip().lower() == "all" else _split(matrix.task)
    attacks = (
        sorted(ATTACK_REGISTRY)
        if matrix.attacks.strip().lower() == "all"
        else [normalize_attack_name(x) for x in _split(matrix.attacks)]
    )
    defenses = (
        sorted(DEFENSE_REGISTRY)
        if matrix.defenses.strip().lower() == "all"
        else [normalize_defense_name(x) for x in _split(matrix.defenses)]
    )
    _validate(tasks, TASK_REGISTRY, "task")
    _validate(attacks, ATTACK_REGISTRY, "attack")
    _validate(defenses, DEFENSE_REGISTRY, "defense")
    for attack in attacks:
        validate_attack_config(attack, base)
    if args.dry_run:
        print(f"Planned runs: {len(tasks) * len(attacks) * len(defenses)}")
        for task_name in tasks:
            for attack in attacks:
                for defense in defenses:
                    print(f"{task_name} {attack} {defense}")
        return
    if matrix.log_dir is None:
        output_dir = Path.cwd() / "log" / "pipeline"
    else:
        output_dir = Path(matrix.log_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent.parent / output_dir
    for task_name in tasks:
        data_cfg = copy.deepcopy(base)
        data_cfg.task_name = task_name
        task = get_task(data_cfg)
        data_cfg.num_classes = task.num_classes
        prepared = task.build_dataloaders(data_cfg)
        for attack in attacks:
            for defense in defenses:
                path = _run_combo(
                    base,
                    task_name,
                    attack,
                    defense,
                    output_dir,
                    prepared,
                    table,
                    {"federated": str(fed_path), "hyperparameters": str(hyper_path)},
                    matrix.fed_config_overrides,
                )
                print(f"Saved: {path}")


if __name__ == "__main__":
    main()
