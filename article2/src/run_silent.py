"""Silent wrapper: suppresses per-round monitor output, only prints progress.

Run from project root::

    python -m src.run_silent --log-dir log/sanity_50r
"""
from __future__ import annotations

import json
from pathlib import Path

from . import main as _main
from .clients import ATTACK_REGISTRY
from .config import FedConfig, project_root
from .run_matrix import apply_defense_to_config, run_one_combo
from .server import DEFENSE_REGISTRY

# Monkey-patch print_monitor_round to be silent
_main.print_monitor_round = lambda **kw: None

def run_silent(
    attacks,
    defenses,
    rounds,
    log_dir,
    seed=42,
    task="cifar10",
    device=None,
    data_root=None,
    local_epochs=None,
    num_workers=None,
):
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(attacks) * len(defenses)
    idx = 0
    for defense in defenses:
        for attack in attacks:
            idx += 1
            cfg = FedConfig()
            cfg.total_rounds = rounds
            cfg.seed = seed
            if device is not None:
                cfg.device = device
            if data_root is not None:
                cfg.data_root = data_root
            if local_epochs is not None:
                cfg.local_epochs = int(local_epochs)
            if num_workers is not None:
                cfg.num_workers = int(num_workers)
            print(f"[{idx}/{total}] {task} | attack={attack} | defense={defense} ...", flush=True)
            out = run_one_combo(cfg, task, attack, defense, output_dir)
            # Quick summary from JSON
            data = json.loads(out.read_text())
            rm = data["round_metrics"]
            last = rm[-1]
            print(f"  -> Round {last['round']}: acc={last['test_acc']:.4f}  TPR={last['malicious_detection_rate']:.4f}  FPR={last['benign_false_positive_rate']:.4f}", flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--attacks", default="all")
    p.add_argument("--defenses", default="avg")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument(
        "--log-dir",
        default=str(project_root() / "log" / "sanity_50r"),
        help="Output directory (default: <project>/log/sanity_50r).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", default="cifar10")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    all_attacks = sorted(ATTACK_REGISTRY.keys())
    all_defenses = sorted(DEFENSE_REGISTRY.keys())
    attacks = all_attacks if args.attacks == "all" else [a.strip() for a in args.attacks.split(",")]
    defenses = all_defenses if args.defenses == "all" else [d.strip() for d in args.defenses.split(",")]

    run_silent(
        attacks,
        defenses,
        args.rounds,
        args.log_dir,
        args.seed,
        args.task,
        device=args.device,
        data_root=args.data_root,
        local_epochs=args.local_epochs,
        num_workers=args.num_workers,
    )
