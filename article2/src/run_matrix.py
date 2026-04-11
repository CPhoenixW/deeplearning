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
        description="Run federated experiments; options align with main.run_federated + FedConfig.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (from project root, the directory containing ``src/``, ``data/``, ``log/``):
  python -m src.run_matrix --list
  python -m src.run_matrix --task cifar10 --attacks bd --defenses mk --rounds 50
  python -m src.run_matrix --task fashion_mnist --attacks gn --defenses avg
  python -m src.run_matrix --task all --attacks all --defenses svdd,avg
""".strip(),
    )
    parser.add_argument("--list", action="store_true", help="Print valid task / attack / defense keys and exit.")
    parser.add_argument("--rounds", type=int, default=300, help="FedConfig.total_rounds")
    parser.add_argument("--num-clients", type=int, default=50, help="FedConfig.num_clients")
    parser.add_argument("--num-benign", type=int, default=35, help="FedConfig.num_benign")
    parser.add_argument("--task", type=str, default="cifar10")
    parser.add_argument("--attacks", type=str, default="all")
    parser.add_argument("--defenses", type=str, default="all")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(project_root() / "log"),
        help="Output JSON directory (default: <project>/log, not cwd-relative).",
    )
    parser.add_argument("--data-root", type=str, default=None, help="FedConfig.data_root")
    parser.add_argument("--local-epochs", type=int, default=1, help="FedConfig.local_epochs")
    parser.add_argument("--num-workers", type=int, default=None, help="FedConfig.num_workers")
    parser.add_argument("--dirichlet-alpha", type=float, default=None, help="FedConfig.dirichlet_alpha")
    parser.add_argument("--seed", type=int, default=42, help="FedConfig.seed")
    parser.add_argument("--device", type=str, default="cuda", choices=("auto", "cuda", "cpu"), help="FedConfig.device")
    parser.add_argument("--trimmed-mean-num-byzantine", type=int, default=None)
    args = parser.parse_args()

    if args.list:
        print_list_options()
        sys.exit(0)

    task_names = parse_tasks_arg(args.task, default_cfg.task_name)

    if args.attacks.strip().lower() == "all":
        attacks = sorted(ATTACK_REGISTRY.keys())
    else:
        attacks = [normalize_attack_name(x) for x in parse_list_arg(args.attacks)]

    defenses = [normalize_defense_name(x) for x in parse_list_arg(args.defenses)]
    validate_lists(task_names, attacks, defenses)

    output_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FedConfig()
    if args.rounds is not None:
        cfg.total_rounds = args.rounds
    if args.num_clients is not None:
        cfg.num_clients = args.num_clients
    if args.num_benign is not None:
        cfg.num_benign = args.num_benign
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.local_epochs is not None:
        cfg.local_epochs = int(args.local_epochs)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = None if args.dirichlet_alpha < 0 else float(args.dirichlet_alpha)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.trimmed_mean_num_byzantine is not None:
        cfg.trimmed_mean_num_byzantine = args.trimmed_mean_num_byzantine

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
