"""Run reproducible federated-learning experiment sweeps.

The script deliberately reuses :func:`src.main.run_federated` rather than
duplicating training logic.  One dataloader partition is prepared per
task/alpha/seed and reused across attacks and defenses, making comparisons
paired and substantially cheaper.

Example (small smoke matrix)::

    python -m src.run_sweep --rounds 5 --clients 10 \
      --attacks none,lf --defenses avg,svdd --rates 0.3 \
      --alphas iid,0.5 --seeds 42 --device cpu --output-dir log/sweep_smoke
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .clients import ATTACK_REGISTRY, mixed_attack_for_client
    from .config import (
        FedConfig,
        normalize_attack_name,
        normalize_defense_name,
        project_root,
    )
    from .main import run_federated
    from .server import DEFENSE_REGISTRY
    from .tasks import TASK_REGISTRY, get_task
except ImportError:  # pragma: no cover - supports ``python src/run_sweep.py``
    from clients import ATTACK_REGISTRY, mixed_attack_for_client
    from config import FedConfig, normalize_attack_name, normalize_defense_name, project_root
    from main import run_federated
    from server import DEFENSE_REGISTRY
    from tasks import TASK_REGISTRY, get_task


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _float_csv(value: str) -> list[float]:
    return [float(x) for x in _csv(value)]


def _int_csv(value: str) -> list[int]:
    return [int(x) for x in _csv(value)]


def _alpha(value: str) -> float | None:
    if value.strip().lower() in {"iid", "none", "inf", "infinity"}:
        return None
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"Dirichlet alpha must be positive or 'iid', got {value!r}.")
    return out


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Tie-aware ROC AUC without scipy/sklearn.

    ``None`` is returned when a round has only one class or non-finite scores.
    """
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))]
    pos = sum(y == 1 for _, y in pairs)
    neg = sum(y == 0 for _, y in pairs)
    if pos == 0 or neg == 0:
        return None
    # Mann-Whitney U with average ranks for ties.
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(ordered)
    i = 0
    rank = 1
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        rank += j - i
        i = j
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Average precision for the malicious (positive) class."""
    pairs = sorted(
        ((float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(y == 1 for _, y in pairs)
    if positives == 0:
        return None
    hits = 0
    area = 0.0
    for rank, (_, label) in enumerate(pairs, 1):
        if label == 1:
            hits += 1
            area += hits / float(rank)
    return float(area / positives)


def _summary(rounds: list[dict[str, Any]], num_benign: int) -> dict[str, Any]:
    if not rounds:
        return {}
    tail = rounds[-min(10, len(rounds)) :]
    def mean(key: str) -> float:
        return float(sum(float(x[key]) for x in tail) / len(tail))

    labels = [0] * (len(rounds[-1].get("selected_mask", [])) - num_benign)
    labels = [0] * num_benign + [1] * max(0, len(rounds[-1].get("selected_mask", [])) - num_benign)
    last = rounds[-1]
    auc = _auc(labels, last.get("detection_scores", [])) if labels else None
    auprc = _auprc(labels, last.get("detection_scores", [])) if labels else None
    return {
        "last": {k: last.get(k) for k in ("round", "test_acc", "malicious_detection_rate", "benign_false_positive_rate", "dpr", "rr", "reject_rate")},
        "mean_last_10": {k: mean(k) for k in ("test_acc", "malicious_detection_rate", "benign_false_positive_rate", "dpr", "rr", "reject_rate")},
        "best_test_acc": max(float(x["test_acc"]) for x in rounds),
        "mean_last_10_backdoor_asr": (
            float(sum(float(x["backdoor_asr"]) for x in tail if x.get("backdoor_asr") is not None) /
                  max(1, sum(x.get("backdoor_asr") is not None for x in tail)))
            if any(x.get("backdoor_asr") is not None for x in tail)
            else None
        ),
        "last_auroc": auc,
        "last_auprc": auprc,
    }


def _set_defense(cfg: FedConfig, defense: str) -> None:
    defense = normalize_defense_name(defense)
    cfg.defense_type = defense
    if defense != "svdd":
        cfg.aggregation_method = defense


def _validate(tasks: Iterable[str], attacks: Iterable[str], defenses: Iterable[str]) -> None:
    for name in tasks:
        if name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASK_REGISTRY)}")
    for name in attacks:
        if name not in ATTACK_REGISTRY:
            raise ValueError(f"Unknown attack {name!r}; choose from {sorted(ATTACK_REGISTRY)}")
    for name in defenses:
        if name not in DEFENSE_REGISTRY:
            raise ValueError(f"Unknown defense {name!r}; choose from {sorted(DEFENSE_REGISTRY)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired federated-learning experiment sweeps.")
    parser.add_argument("--tasks", default="cifar10")
    parser.add_argument("--attacks", default="none,lf,gn,sf,lie,bd,mix")
    parser.add_argument(
        "--mixed-attack-types",
        default="lf,bd,gn",
        help="Comma-separated attack IDs assigned round-robin to malicious clients when --attacks includes mix.",
    )
    parser.add_argument("--defenses", default="avg,tm,mk,svdd")
    parser.add_argument("--clients", default="10")
    parser.add_argument("--rates", default="0.1,0.2,0.3")
    parser.add_argument("--alphas", default="iid,0.1,0.5,1.0")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--phase1-rounds", type=int, default=15)
    parser.add_argument("--svdd-recon-lambda", type=float, default=0.1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--phase1-selection",
        choices=("reconstruction", "feature_median"),
        default="reconstruction",
        help="Phase-1 selector; feature_median is a legacy ablation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "log" / "sweep"))
    parser.add_argument("--dry-run", action="store_true", help="Print combinations without training.")
    args = parser.parse_args()

    tasks = [x.lower() for x in _csv(args.tasks)]
    attacks = [normalize_attack_name(x.lower()) for x in _csv(args.attacks)]
    defenses = [normalize_defense_name(x.lower()) for x in _csv(args.defenses)]
    clients = _int_csv(args.clients)
    rates = _float_csv(args.rates)
    alphas = [_alpha(x) for x in _csv(args.alphas)]
    seeds = _int_csv(args.seeds)
    _validate(tasks, attacks, defenses)
    if any(k < 2 for k in clients):
        raise ValueError("Each client count must be at least 2.")
    if any(r < 0.0 or r >= 1.0 for r in rates):
        raise ValueError("Attack rates must satisfy 0 <= rate < 1.")

    combos = list(itertools.product(tasks, attacks, defenses, clients, rates, alphas, seeds))
    print(f"Planned runs: {len(combos)}")
    if args.dry_run:
        for combo in combos:
            print(" ".join(map(str, combo)))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = FedConfig(
        total_rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        device=args.device,
    )
    base.phase1_rounds = int(args.phase1_rounds)
    base.svdd_recon_lambda = float(args.svdd_recon_lambda)
    if args.data_root is not None:
        base.data_root = args.data_root
    base.round_diagnostics = False
    base.reuse_client_model = True
    base.skip_redundant_attack_training = True
    base.phase1_selection = args.phase1_selection
    base.mixed_attack_types = args.mixed_attack_types

    # Cache partitions by task/clients/alpha/seed.  Attacks and defenses share
    # exactly the same client data and hence produce paired comparisons.
    loader_cache: dict[tuple[str, int, float | None, int], Any] = {}
    for idx, (task_name, attack, defense, k, rate, alpha, seed) in enumerate(combos, 1):
        effective_rate = 0.0 if attack == "none" else rate
        num_malicious = int(round(k * effective_rate))
        num_benign = k - num_malicious
        if num_benign < 1:
            raise ValueError("Every run must retain at least one benign client.")
        key = (task_name, k, alpha, seed)
        if key not in loader_cache:
            data_cfg = copy.deepcopy(base)
            data_cfg.task_name = task_name
            data_cfg.num_clients = k
            data_cfg.num_benign = num_benign
            data_cfg.seed = seed
            data_cfg.dirichlet_alpha = alpha
            loader_cache[key] = get_task(data_cfg).build_dataloaders(data_cfg)

        cfg = copy.deepcopy(base)
        cfg.task_name, cfg.attack_type, cfg.num_clients = task_name, attack, k
        cfg.num_benign, cfg.seed, cfg.dirichlet_alpha = num_benign, seed, alpha
        _set_defense(cfg, defense)
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"[{idx}/{len(combos)}] {task_name} attack={attack} defense={defense} K={k} rate={rate} alpha={alpha} seed={seed}", flush=True)
        rounds = run_federated(cfg, collect_metrics=True, prepared_dataloaders=loader_cache[key]) or []
        payload = {
            "meta": {
                "task_name": task_name, "attack": attack, "defense": defense,
                "num_clients": k, "num_benign": num_benign,
                "malicious_rate": effective_rate, "requested_rate": rate, "dirichlet_alpha": alpha,
                "mixed_attack_types": cfg.mixed_attack_types,
                "mixed_attack_assignments": (
                    {str(cid): mixed_attack_for_client(cfg, cid) for cid in range(cfg.num_benign, cfg.num_clients)}
                    if attack == "mix" else {}
                ),
                "seed": seed, "total_rounds": args.rounds,
                "device": args.device, "started_at": started,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            "summary": _summary(rounds, num_benign),
            "round_metrics": rounds,
        }
        alpha_tag = "iid" if alpha is None else f"{alpha:g}"
        stem = f"{task_name}__{attack}__{defense}__K{k}__r{rate:g}__a{alpha_tag}__s{seed}"
        (output_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
