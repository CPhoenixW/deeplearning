from __future__ import annotations
import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .clients import ATTACK_REGISTRY
    from .config import (
        ATTACK_ALIASES,
        DEFENSE_ALIASES,
        FedConfig,
        load_matrix_run_config,
        normalize_attack_name,
        normalize_defense_name,
        project_root,
    )
    from .main import run_federated
    from .server import DEFENSE_REGISTRY
    from .tasks import TASK_REGISTRY
except ImportError:
    from clients import ATTACK_REGISTRY
    from config import (
        ATTACK_ALIASES,
        DEFENSE_ALIASES,
        FedConfig,
        load_matrix_run_config,
        normalize_attack_name,
        normalize_defense_name,
        project_root,
    )
    from main import run_federated
    from server import DEFENSE_REGISTRY
    from tasks import TASK_REGISTRY


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_tasks_arg(value: str | None, default_task_name: str) -> List[str]:
    """``all`` → 全部任务；逗号分隔；``None`` → 仅默认任务名。"""
    if value is None or value.strip() == "":
        return [default_task_name]
    v = value.strip().lower()
    if v == "all":
        return sorted(TASK_REGISTRY.keys())
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    return names


def validate_choice(name: str, value: str, registry: Dict[str, object]) -> None:
    if value not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown {name} {value!r}. Available: {available}")


def validate_lists(
    tasks: Sequence[str],
    attacks: Sequence[str],
    defenses: Sequence[str],
) -> None:
    for t in tasks:
        validate_choice("task_name", t, TASK_REGISTRY)
    for a in attacks:
        validate_choice("attack_type", a, ATTACK_REGISTRY)
    for d in defenses:
        validate_choice("defense_type", d, DEFENSE_REGISTRY)


def apply_defense_to_config(cfg: FedConfig, defense: str) -> None:
    """与 ``main.resolve_defense_name(..., use_svdd=None)`` 一致：只看 ``defense_type``。"""
    d = normalize_defense_name(defense)
    cfg.defense_type = d
    if d != "svdd":
        cfg.aggregation_method = d


def run_one_combo(
    base_cfg: FedConfig,
    task_name: str,
    attack: str,
    defense: str,
    output_dir: Path,
) -> Path:
    attack_id = normalize_attack_name(attack)
    defense_id = normalize_defense_name(defense)
    cfg = copy.deepcopy(base_cfg)
    cfg.task_name = task_name
    cfg.attack_type = attack_id
    apply_defense_to_config(cfg, defense_id)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rounds = run_federated(cfg, use_svdd=None, collect_metrics=True)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    if rounds is None:
        rounds = []

    payload: Dict[str, object] = {
        "meta": {
            "task_name": cfg.task_name,
            "attack": attack_id,
            "defense": defense_id,
            "num_classes": cfg.num_classes,
            "started_at": started_at,
            "finished_at": finished_at,
            "num_clients": cfg.num_clients,
            "num_benign": cfg.num_benign,
            "total_rounds": cfg.total_rounds,
            "defense_type": cfg.defense_type,
            "aggregation_method": cfg.aggregation_method,
            "data_root": cfg.data_root,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_noniid_beta": cfg.dirichlet_noniid_beta,
            "seed": cfg.seed,
            "device": cfg.device,
            "trimmed_mean_ratio": cfg.trimmed_mean_ratio,
            "trimmed_mean_num_byzantine": cfg.trimmed_mean_num_byzantine,
            "krum_num_byzantine": cfg.krum_num_byzantine,
            "multi_krum_num_selected": cfg.multi_krum_num_selected,
        },
        "round_metrics": rounds,
    }

    filename = f"{task_name}__{attack_id}__{defense_id}.json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_list_options() -> None:
    def _aka_short_to_long(aliases: Dict[str, str]) -> Dict[str, list[str]]:
        out: Dict[str, list[str]] = {}
        for long_n, short_n in aliases.items():
            out.setdefault(short_n, []).append(long_n)
        return out

    atk_old = _aka_short_to_long(ATTACK_ALIASES)
    def_old = _aka_short_to_long(DEFENSE_ALIASES)

    print("Tasks (dataset + model, config.task_name):")
    for k in sorted(TASK_REGISTRY.keys()):
        print(f"  - {k}")
    print("Attacks (short id; long names still accepted):")
    for k in sorted(ATTACK_REGISTRY.keys()):
        olds = atk_old.get(k, [])
        extra = f"  (aka {', '.join(olds)})" if olds else ""
        print(f"  - {k}{extra}")
    print("Defenses (short id; long names still accepted):")
    for k in sorted(DEFENSE_REGISTRY.keys()):
        olds = def_old.get(k, [])
        extra = f"  (aka {', '.join(olds)})" if olds else ""
        print(f"  - {k}{extra}")


def main() -> None:
    default_cfg = FedConfig()
    parser = argparse.ArgumentParser(
        description="按 ``config.MatrixRunConfig`` / ``load_matrix_run_config`` 跑任务×攻击×防御矩阵。",
    )
    parser.add_argument("--list", action="store_true", help="列出合法 task / attack / defense 键后退出。")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 文件，键为 MatrixRunConfig 字段，用于覆盖 ``config.DEFAULT_MATRIX_RUN``。",
    )
    args = parser.parse_args()

    if args.list:
        print_list_options()
        sys.exit(0)

    mr = load_matrix_run_config(args.config)

    task_names = parse_tasks_arg(mr.task, default_cfg.task_name)

    if mr.attacks.strip().lower() == "all":
        attacks = sorted(ATTACK_REGISTRY.keys())
    else:
        attacks = [normalize_attack_name(x) for x in parse_list_arg(mr.attacks)]

    defenses = [normalize_defense_name(x) for x in parse_list_arg(mr.defenses)]
    validate_lists(task_names, attacks, defenses)

    log_dir = mr.log_dir if mr.log_dir is not None else str(project_root() / "log")
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FedConfig()
    cfg.total_rounds = mr.total_rounds
    cfg.num_clients = mr.num_clients
    cfg.num_benign = mr.num_benign
    if mr.data_root is not None:
        cfg.data_root = mr.data_root
    cfg.local_epochs = int(mr.local_epochs)
    if mr.num_workers is not None:
        cfg.num_workers = int(mr.num_workers)
    if mr.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = None if mr.dirichlet_alpha < 0 else float(mr.dirichlet_alpha)
    cfg.seed = mr.seed
    cfg.device = mr.device
    if mr.trimmed_mean_num_byzantine is not None:
        cfg.trimmed_mean_num_byzantine = mr.trimmed_mean_num_byzantine

    print(f"Tasks: {task_names}")
    print(f"Attacks: {attacks}")
    print(f"Defenses: {defenses}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"data_root: {cfg.data_root}")
    print(f"dirichlet_alpha: {cfg.dirichlet_alpha}")

    total = len(task_names) * len(attacks) * len(defenses)
    idx = 0
    for task_name in task_names:
        for attack in attacks:
            for defense in defenses:
                idx += 1
                print(f"\n=== [{idx}/{total}] task={task_name} attack={attack} defense={defense} ===")
                out_path = run_one_combo(cfg, task_name, attack, defense, output_dir)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
