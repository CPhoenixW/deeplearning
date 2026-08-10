#!/usr/bin/env python3
"""Generate the confirmed 624-job Article2 primary experiment matrix.

This command only writes independent JSON configs and a manifest.  It does not
launch training, so the generated matrix can be reviewed before scheduling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TASKS = ("mnist", "fashion_mnist", "cifar10")
IMAGE_ATTACKS = ("none", "lf", "gn", "sf", "lie", "bd", "mix")
AG_NEWS_ATTACKS = ("none", "lf", "gn", "sf", "lie")
DEFENSES = ("avg", "tm", "mk", "lasa", "seca", "bnguard", "dmc", "svdd")
SEEDS = (42, 43, 44)

BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
    "server_validation_size": 50,
    "local_epochs": 1,
    "batch_size": 64,
    "num_workers": 0,
    "dirichlet_alpha": 1.0,
    "phase1_rounds": 15,
    "phase1_score_mode": "recon",
    "phase2_score_mode": "svdd",
    "alpha": 0.5,
    "lie_z_override": 0.524,
    "mixed_attack_types": "lf,bd,gn",
    "param_descriptor_dim": 4096,
    "param_descriptor_device": "cuda",
    "device": "cuda",
}


@dataclass(frozen=True)
class MatrixJob:
    task: str
    attack: str
    defense: str
    seed: int
    config_path: str
    log_dir: str


def iter_specs() -> Iterable[tuple[str, str, str, int]]:
    for task in IMAGE_TASKS + ("ag_news",):
        attacks = IMAGE_ATTACKS if task != "ag_news" else AG_NEWS_ATTACKS
        for attack in attacks:
            for defense in DEFENSES:
                for seed in SEEDS:
                    yield task, attack, defense, seed


def write_matrix(root: Path, *, rounds: int = 300, force: bool = False) -> list[MatrixJob]:
    root = root.resolve()
    configs_root = root / "_configs"
    jobs: list[MatrixJob] = []
    for task, attack, defense, seed in iter_specs():
        relative = Path(task) / attack / defense / f"seed_{seed}"
        log_dir = root / relative
        config_path = configs_root / relative.with_suffix(".json")
        if config_path.exists() and not force:
            jobs.append(MatrixJob(task, attack, defense, seed, str(config_path), str(log_dir)))
            continue
        overrides = dict(BASE_OVERRIDES)
        overrides.update({"seed": seed, "total_rounds": int(rounds)})
        payload = {
            "task": task,
            "attacks": attack,
            "defenses": defense,
            "log_dir": str(log_dir),
            "fed_config_file": "configs/federated.json",
            "hyperparameters_file": "configs/hyperparameters.json",
            "fed_config_overrides": overrides,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        jobs.append(MatrixJob(task, attack, defense, seed, str(config_path), str(log_dir)))
    manifest = {
        "protocol": "article2-primary-v1",
        "rounds": int(rounds),
        "tasks": list(IMAGE_TASKS + ("ag_news",)),
        "image_attacks": list(IMAGE_ATTACKS),
        "ag_news_attacks": list(AG_NEWS_ATTACKS),
        "defenses": list(DEFENSES),
        "seeds": list(SEEDS),
        "jobs": [asdict(job) for job in jobs],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("log/article2_primary_624"))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    jobs = write_matrix(args.output_root, rounds=args.rounds, force=args.force)
    expected = (3 * 7 + 5) * 8 * 3
    print(f"jobs={len(jobs)} expected={expected} rounds={args.rounds}")
    if len(jobs) != expected:
        raise RuntimeError(f"Primary matrix cardinality mismatch: {len(jobs)} != {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
