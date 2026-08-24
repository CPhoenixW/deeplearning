#!/usr/bin/env python3
"""Run the 4096-dim absolute-parameter SVDD Fashion-MNIST ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


DEFAULT_ATTACKS = ("gn", "sf", "lf", "bd")
DEFAULT_SEEDS = (42, 43, 44)

BASE_OVERRIDES = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 100,
    "client_lr": 0.1,
    "client_momentum": 0.9,
    "client_weight_decay": 0.0,
    "client_grad_clip": 5.0,
    "client_update_clip": 5.0,
    "local_epochs": 1,
    "batch_size": 64,
    "num_workers": 0,
    "use_amp": False,
    "channels_last": False,
    "cuda_aggregation": True,
    "reuse_client_model": True,
    "skip_redundant_attack_training": True,
    "client_batch_group_size": 2,
    "round_diagnostics": False,
    "dirichlet_alpha": 1.0,
    "latent_dim": 64,
    "ae_lr": 0.001,
    "ae_weight_decay": 1e-6,
    "ae_grad_clip": 1.0,
    "svdd_input_mode": "absolute",
    "svdd_input_dim": 4096,
    "svdd_normalization_eps": 1e-6,
    "phase1_rounds": 15,
    "phase1_score_mode": "recon",
    "phase2_score_mode": "combined",
    "svdd_lambda": 0.5,
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "device": "cuda",
}


def _csv_values(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _paths(root: Path, attack: str, seed: int) -> tuple[Path, Path, Path]:
    output_dir = root / attack / f"seed_{seed}"
    config_path = root / "_configs" / attack / f"seed_{seed}.json"
    result_path = output_dir / f"fashion_mnist__{attack}__svdd.json"
    return config_path, output_dir, result_path


def _write_config(root: Path, attack: str, seed: int, rounds: int, malicious: int) -> tuple[Path, Path, Path]:
    config_path, output_dir, result_path = _paths(root, attack, seed)
    overrides = dict(BASE_OVERRIDES)
    overrides.update(
        {
            "seed": int(seed),
            "total_rounds": int(rounds),
            "num_malicious": int(malicious),
        }
    )
    payload = {
        "task": "fashion_mnist",
        "attacks": attack,
        "defenses": "svdd",
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path, output_dir, result_path


def _complete(result_path: Path, rounds: int) -> bool:
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return len(payload.get("rounds", [])) == int(rounds)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _run_one(python_bin: str, script: Path, config_path: Path, output_dir: Path, gpu: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    console_path = output_dir / "console.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [python_bin, "-u", str(script), "--config", str(config_path)]
    print("RUN", " ".join(command), f"CUDA_VISIBLE_DEVICES={gpu}", flush=True)
    with console_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(script.parent),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print("DONE", config_path, f"returncode={completed.returncode}", flush=True)
    return int(completed.returncode)


def _summary(root: Path, attacks: tuple[str, ...], seeds: tuple[int, ...], rounds: int) -> Path:
    rows = []
    for attack in attacks:
        for seed in seeds:
            _, _, result_path = _paths(root, attack, seed)
            if not _complete(result_path, rounds):
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            final = payload["rounds"][-10:]
            values = {"attack": attack, "seed": seed, "rounds": len(payload["rounds"])}
            for key in ("accuracy", "dar", "tpr", "fpr", "rr", "reject_rate", "backdoor_asr"):
                series = [
                    float(item["evaluation"][key])
                    for item in final
                    if item.get("evaluation", {}).get(key) is not None
                ]
                if series:
                    values[f"{key}_mean10"] = statistics.mean(series)
            rows.append(values)
    output = root / "summary_by_run.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--malicious", type=int, default=30)
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent runs assigned to each listed GPU; keep at 1 unless memory permits more.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("log/fashion_svdd_absolute_4096"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    attacks = _csv_values(args.attacks)
    seeds = _csv_values(args.seeds, int)
    gpus = _csv_values(args.gpus)
    if not attacks or not seeds or not gpus:
        parser.error("attacks, seeds, and gpus must not be empty")
    if args.rounds < 2 or args.malicious < 0 or args.malicious >= 100:
        parser.error("rounds must be >= 2 and malicious must be in [0, 99]")
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be positive")
    visible_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    invalid_gpus = [gpu for gpu in gpus if not str(gpu).isdigit() or int(gpu) >= visible_gpus]
    if invalid_gpus:
        parser.error(
            f"requested GPU ids {invalid_gpus}, but this process can see {visible_gpus} CUDA device(s)"
        )

    root = args.output_root.resolve()
    script = (Path(__file__).resolve().parents[1] / "svdd_test.py").resolve()
    jobs = []
    for seed in seeds:
        for attack in attacks:
            config_path, output_dir, result_path = _write_config(
                root, attack, seed, args.rounds, args.malicious
            )
            if args.force or not _complete(result_path, args.rounds):
                jobs.append((attack, seed, config_path, output_dir, result_path))
    if args.max_jobs is not None:
        jobs = jobs[: max(0, int(args.max_jobs))]

    print(
        f"planned={len(jobs)} attacks={attacks} seeds={seeds} rounds={args.rounds} "
        f"malicious={args.malicious}% input_dim=4096 gpus={gpus} "
        f"workers_per_gpu={args.workers_per_gpu}"
    )
    for attack, seed, config_path, _, _ in jobs:
        print(f"PENDING attack={attack} seed={seed} config={config_path}")
    if args.dry_run:
        return 0

    failures = []
    slots = [gpu for gpu in gpus for _ in range(args.workers_per_gpu)]
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = {
            executor.submit(
                _run_one,
                str(args.python),
                script,
                config_path,
                output_dir,
                str(slots[index % len(slots)]),
            ): (attack, seed)
            for index, (attack, seed, config_path, output_dir, _) in enumerate(jobs)
        }
        for future in as_completed(futures):
            attack, seed = futures[future]
            try:
                code = int(future.result())
            except Exception as exc:  # pragma: no cover - defensive launcher guard
                print(f"LAUNCHER_ERROR attack={attack} seed={seed}: {exc}", flush=True)
                code = 1
            if code:
                failures.append((attack, seed, code))
    summary_path = _summary(root, attacks, seeds, args.rounds)
    print(f"SUMMARY {summary_path}")
    if failures:
        print(f"FAILURES {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
