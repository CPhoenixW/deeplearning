from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from .clients import ATTACK_REGISTRY
    from .config import FedConfig
    from .main import run_federated
except ImportError:
    from clients import ATTACK_REGISTRY
    from config import FedConfig
    from main import run_federated


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_one_combo(
    base_cfg: FedConfig,
    attack: str,
    defense: str,
    output_dir: Path,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.attack_type = attack
    cfg.defense_type = defense
    if defense != "svdd":
        cfg.aggregation_method = defense

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rounds = run_federated(cfg, use_svdd=None, collect_metrics=True)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    if rounds is None:
        rounds = []

    payload: Dict[str, object] = {
        "meta": {
            "attack": attack,
            "defense": defense,
            "task_name": cfg.task_name,
            "num_classes": cfg.num_classes,
            "started_at": started_at,
            "finished_at": finished_at,
            "num_clients": cfg.num_clients,
            "num_benign": cfg.num_benign,
            "total_rounds": cfg.total_rounds,
            "defense_type": cfg.defense_type,
            "use_svdd": cfg.defense_type == "svdd",
            "aggregation_method": cfg.aggregation_method if cfg.defense_type != "svdd" else "svdd",
            "trimmed_mean_ratio": cfg.trimmed_mean_ratio,
            "krum_num_byzantine": cfg.krum_num_byzantine,
            "multi_krum_num_selected": cfg.multi_krum_num_selected,
        },
        "round_metrics": rounds,
    }

    filename = f"{attack}__{defense}.json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attack-defense matrix and dump JSON logs.")
    parser.add_argument("--rounds", type=int, default=None, help="Override total rounds.")
    parser.add_argument("--num-clients", type=int, default=None, help="Override number of clients.")
    parser.add_argument("--num-benign", type=int, default=None, help="Override number of benign clients.")
    parser.add_argument(
        "--attacks",
        type=str,
        default="all",
        help="Comma-separated attacks or 'all'.",
    )
    parser.add_argument(
        "--defenses",
        type=str,
        default="svdd,fedavg,trimmed_mean,multi_krum",
        help="Comma-separated defenses.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="Directory for JSON logs.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (dataset+model), e.g. cifar10 or fashion_mnist.",
    )
    args = parser.parse_args()

    output_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FedConfig()
    if args.task is not None:
        cfg.task_name = args.task
    if args.rounds is not None:
        cfg.total_rounds = args.rounds
    if args.num_clients is not None:
        cfg.num_clients = args.num_clients
    if args.num_benign is not None:
        cfg.num_benign = args.num_benign

    if args.attacks == "all":
        attacks = sorted(ATTACK_REGISTRY.keys())
    else:
        attacks = parse_list_arg(args.attacks)

    defenses = parse_list_arg(args.defenses)
    valid_defenses = {"svdd", "fedavg", "trimmed_mean", "multi_krum"}
    invalid_defenses = [d for d in defenses if d not in valid_defenses]
    if invalid_defenses:
        raise ValueError(f"Invalid defenses: {invalid_defenses}")

    invalid_attacks = [a for a in attacks if a not in ATTACK_REGISTRY]
    if invalid_attacks:
        raise ValueError(f"Invalid attacks: {invalid_attacks}")

    print(f"Attacks: {attacks}")
    print(f"Defenses: {defenses}")
    print(f"Output dir: {output_dir.resolve()}")

    for attack in attacks:
        for defense in defenses:
            print(f"\n=== Running attack={attack}, defense={defense} ===")
            out_path = run_one_combo(cfg, attack, defense, output_dir)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

