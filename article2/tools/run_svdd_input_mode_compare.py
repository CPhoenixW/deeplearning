#!/usr/bin/env python3
"""Compare absolute-weight and delta-weight 4096-D descriptor SVDD inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MODES = ("absolute", "delta")

BASE_OVERRIDES = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
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
    "svdd_input_dim": 4096,
    "svdd_normalization": "mean_std",
    "svdd_normalization_eps": 1e-6,
    "svdd_descriptor_device": "cuda",
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


def _paths(root: Path, mode: str, seed: int) -> tuple[Path, Path, Path]:
    output = root / mode / f"seed_{seed}"
    config = root / "_configs" / mode / f"seed_{seed}.json"
    result = output / f"fashion_mnist__bd__svdd.json"
    return config, output, result


def _write_config(root: Path, mode: str, seed: int, rounds: int) -> tuple[Path, Path, Path]:
    config, output, result = _paths(root, mode, seed)
    overrides = dict(BASE_OVERRIDES)
    overrides.update({"seed": seed, "total_rounds": rounds, "svdd_input_mode": mode})
    payload = {
        "task": "fashion_mnist",
        "attacks": "bd",
        "defenses": "svdd",
        "log_dir": str(output),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config, output, result


def _run_one(python_bin: str, script: Path, config: Path, output: Path, gpu: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    command = [python_bin, "-u", str(script), "--config", str(config)]
    with (output / "console.log").open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=str(script.parent), env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def _summarize(
    root: Path, seeds: tuple[int, ...], rounds: int, modes: tuple[str, ...]
) -> Path:
    rows = []
    for mode in modes:
        for seed in seeds:
            _, _, result = _paths(root, mode, seed)
            if not result.exists():
                continue
            payload = json.loads(result.read_text(encoding="utf-8"))
            tail = payload.get("rounds", [])[-10:]
            values = {"mode": mode, "seed": seed, "rounds": len(payload.get("rounds", []))}
            for key in ("accuracy", "dar", "rr", "backdoor_asr"):
                series = [float(item["evaluation"][key]) for item in tail if item.get("evaluation", {}).get(key) is not None]
                if series:
                    values[f"{key}_mean10"] = statistics.mean(series)
            rows.append(values)
    output = root / "summary_by_run.csv"
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--modes", default=','.join(MODES))
    parser.add_argument("--output-root", type=Path, default=Path("log/fashion_svdd_input_compare_bd_3gpu"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if not seeds or args.rounds < 2:
        parser.error("seeds must be non-empty and rounds must be at least 2")
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus or args.workers_per_gpu < 1:
        parser.error("gpus must be non-empty and workers-per-gpu must be positive")
    modes = tuple(item.strip().lower() for item in args.modes.split(",") if item.strip())
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"modes must be a subset of {MODES}")

    root = args.output_root.resolve()
    script = (Path(__file__).resolve().parents[1] / "svdd_test.py").resolve()
    jobs = [_write_config(root, mode, seed, args.rounds) for mode in modes for seed in seeds]
    worker_count = len(gpus) * args.workers_per_gpu
    assigned = [
        (config, output, gpu)
        for index, (config, output, _result) in enumerate(jobs)
        for gpu in (gpus[(index // args.workers_per_gpu) % len(gpus)],)
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_one, str(args.python), script, config, output, gpu)
            for config, output, gpu in assigned
        ]
        codes = [future.result() for future in futures]
    summary = _summarize(root, seeds, args.rounds, modes)
    print(f"summary={summary} modes={modes} workers={worker_count} gpus={gpus}")
    return 1 if any(codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
