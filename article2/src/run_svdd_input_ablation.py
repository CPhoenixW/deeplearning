"""Run a controlled AE-SVDD input ablation: absolute model vs model delta.

Example from the project root::

    ./dl/bin/python -m src.run_svdd_input_ablation \
        --task cifar10 --attacks gn,lie,bd --rounds 300 --phase1-rounds 15

Both modes reuse the same seed, data partition, malicious-client identities,
client order, and hyperparameters. Results are written to separate JSON files
plus one comparison summary per attack. Per-round results are printed by
default, like ``run_matrix``; pass ``--silent`` to suppress them.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from . import main as federated_main
from .clients import ATTACK_REGISTRY
from .config import FedConfig, normalize_attack_name, project_root


ATTACK_LABELS = {
    "gn": "Gaussian Noise",
    "lf": "Label Flipping",
    "sf": "Sign Flipping",
    "bd": "Backdoor",
    "lie": "LIE / ALIE",
}


def _last_round_summary(rounds: list[dict[str, Any]]) -> dict[str, float]:
    last = rounds[-1]
    return {
        "test_acc": float(last["test_acc"]),
        "tpr": float(last["malicious_detection_rate"]),
        "fpr": float(last["benign_false_positive_rate"]),
        "reject_rate": float(last["reject_rate"]),
    }


def _max_round_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
) -> float:
    if len(left) != len(right):
        raise ValueError(f"Round count mismatch: {len(left)} vs {len(right)}.")
    return max(
        (abs(float(a[key]) - float(b[key])) for a, b in zip(left, right)),
        default=0.0,
    )


def _parse_attacks(value: str) -> list[str]:
    raw = value.strip().lower()
    if raw == "all":
        return sorted(ATTACK_REGISTRY.keys())
    attacks = [
        normalize_attack_name(item)
        for item in value.split(",")
        if item.strip()
    ]
    if not attacks:
        raise ValueError("At least one attack must be specified.")
    unknown = [attack for attack in attacks if attack not in ATTACK_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown attacks {unknown}. Available: {sorted(ATTACK_REGISTRY.keys())}."
        )
    return attacks


def run_mode(
    base: FedConfig,
    attack: str,
    mode: str,
) -> tuple[list[dict[str, Any]], float]:
    cfg = FedConfig(**vars(base))
    cfg.attack_type = attack
    cfg.svdd_input_mode = mode

    started = time.perf_counter()
    rounds = federated_main.run_federated(
        cfg,
        use_svdd=None,
        collect_metrics=True,
    )
    elapsed = time.perf_counter() - started
    if rounds is None:
        raise RuntimeError("Expected collected metrics, got None.")
    return rounds, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare absolute-model and model-delta inputs for AE-SVDD."
    )
    parser.add_argument("--task", default="fashion_mnist")
    parser.add_argument(
        "--attacks",
        default="gn",
        help=(
            "Comma-separated attack IDs or 'all'. "
            "Available: gn, lf, sf, bd, lie."
        ),
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="Backward-compatible single-attack alias; overrides --attacks.",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--phase1-rounds", type=int, default=2)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-benign", type=int, default=8)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "log" / "svdd_input_ablation_smoke"),
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress per-round monitor tables; they are shown by default.",
    )
    parser.add_argument("--show-rounds", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rounds <= args.phase1_rounds:
        raise ValueError("--rounds must be greater than --phase1-rounds.")
    if not 0 <= args.num_benign <= args.num_clients:
        raise ValueError("--num-benign must be in [0, --num-clients].")

    attack_arg = args.attack if args.attack is not None else args.attacks
    attacks = _parse_attacks(attack_arg)
    modes = ("absolute", "delta")
    defense = "svdd"

    base = FedConfig(
        task_name=args.task,
        attack_type=attacks[0],
        defense_type=defense,
        aggregation_method="avg",
        total_rounds=args.rounds,
        phase1_rounds=args.phase1_rounds,
        num_clients=args.num_clients,
        num_benign=args.num_benign,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        device=args.device,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.silent:
        federated_main.print_monitor_round = lambda **_kwargs: None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AE-SVDD Input Ablation")
    print(f"Task             : {args.task}")
    print(
        "Attacks          : "
        + ", ".join(
            f"{attack} ({ATTACK_LABELS.get(attack, attack)})"
            for attack in attacks
        )
    )
    print("Defense          : svdd (AE-SVDD client filtering)")
    print("Input modes      : absolute, delta")
    print(f"Clients          : {args.num_clients} total / {args.num_benign} benign")
    print(f"Dirichlet alpha  : {args.dirichlet_alpha}")
    print(f"Rounds           : {args.rounds} total / {args.phase1_rounds} phase-1")
    print(f"Seed             : {args.seed}")
    print(f"Per-round output : {'hidden' if args.silent else 'shown'}")
    print(f"Output directory : {output_dir.resolve()}")
    print("=" * 72, flush=True)

    all_comparisons: dict[str, dict[str, Any]] = {}
    total_runs = len(attacks) * len(modes)
    run_index = 0

    for attack in attacks:
        print(
            f"\n### Attack: {attack} ({ATTACK_LABELS.get(attack, attack)}) "
            f"| Defense: svdd ###",
            flush=True,
        )
        results: dict[str, list[dict[str, Any]]] = {}
        elapsed_seconds: dict[str, float] = {}

        for mode in modes:
            run_index += 1
            print(
                f"\n=== [{run_index}/{total_runs}] "
                f"task={args.task} attack={attack} defense=svdd "
                f"input={mode} ===",
                flush=True,
            )
            rounds, elapsed = run_mode(base, attack, mode)
            results[mode] = rounds
            elapsed_seconds[mode] = elapsed
            payload = {
                "meta": {
                    **vars(base),
                    "attack_type": attack,
                    "defense_type": defense,
                    "svdd_input_mode": mode,
                    "elapsed_seconds": elapsed,
                },
                "round_metrics": rounds,
            }
            path = output_dir / f"{args.task}__{attack}__svdd__{mode}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"Final: {_last_round_summary(rounds)}\n"
                f"Elapsed: {elapsed:.2f}s\n"
                f"Saved: {path}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        comparison = {
            "configuration": {
                **vars(base),
                "attack_type": attack,
                "defense_type": defense,
                "modes": list(modes),
            },
            "absolute_final": _last_round_summary(results["absolute"]),
            "delta_final": _last_round_summary(results["delta"]),
            "elapsed_seconds": elapsed_seconds,
            "max_per_round_absolute_difference": {
                "test_acc": _max_round_difference(
                    results["absolute"], results["delta"], "test_acc"
                ),
                "tpr": _max_round_difference(
                    results["absolute"],
                    results["delta"],
                    "malicious_detection_rate",
                ),
                "fpr": _max_round_difference(
                    results["absolute"],
                    results["delta"],
                    "benign_false_positive_rate",
                ),
                "reject_rate": _max_round_difference(
                    results["absolute"], results["delta"], "reject_rate"
                ),
            },
            "interpretation": (
                "With per-round coordinate-wise median/MAD centering and a "
                "linear parameter extractor, absolute and delta features are "
                "translation-equivalent in exact arithmetic. Any observed "
                "difference should therefore be checked against floating-point "
                "and repeated-run nondeterminism."
            ),
        }
        all_comparisons[attack] = comparison
        summary_path = (
            output_dir / f"{args.task}__{attack}__svdd__comparison.json"
        )
        summary_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nComparison summary:")
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved comparison: {summary_path}", flush=True)

    index_path = output_dir / f"{args.task}__svdd__all_comparisons.json"
    index_path.write_text(
        json.dumps(all_comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved all comparisons: {index_path}", flush=True)


if __name__ == "__main__":
    main()
