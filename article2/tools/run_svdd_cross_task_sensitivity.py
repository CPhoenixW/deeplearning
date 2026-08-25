#!/usr/bin/env python3
"""Run the additive cross-task SVDD sensitivity matrix.

Each job changes one factor while keeping the other factors at the baseline:
malicious ratio=.30, phase-1 rounds=15, validation/trust size=50, latent=64.
The runner is resumable and records both valid and intentionally skipped jobs
in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TASKS = ("mnist", "fashion_mnist", "cifar10")
ALL_TASKS = IMAGE_TASKS + ("ag_news",)
IMAGE_ATTACKS = ("gn", "sf", "lf", "bd", "lie")
AG_NEWS_ATTACKS = ("gn", "sf", "lf", "lie")
SEEDS = (42, 43, 44)

BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
    "server_validation_size": 50,
    "local_epochs": 1,
    "batch_size": 64,
    "num_workers": 0,
    "use_amp": False,
    "channels_last": False,
    "cuda_aggregation": True,
    "reuse_client_model": True,
    "skip_redundant_attack_training": True,
    "client_batch_group_size": 1,
    "round_diagnostics": False,
    "dirichlet_alpha": 1.0,
    "hf_datasets_offline": True,
    "svdd_input_mode": "absolute",
    "svdd_input_dim": 4096,
    "svdd_normalization": "median_mad",
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


def _factor_specs() -> list[tuple[str, str, dict[str, Any]]]:
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for value in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        label = f"malicious_ratio_{value:.1f}".replace(".", "p")
        specs.append(("malicious_ratio", label, {"num_malicious": int(round(100 * value))}))
    for value in (5, 15, 30, 50, 100):
        specs.append(("phase1_rounds", f"phase1_{value:03d}", {"phase1_rounds": value}))
    for value in (10, 25, 50, 100, 200):
        specs.append(("trust_size", f"trust_{value:03d}", {"server_validation_size": value}))
    for value in (16, 32, 64, 128):
        specs.append(("latent_dim", f"latent_{value:03d}", {"latent_dim": value}))
    return specs


def _result_path(output_dir: Path, task: str, attack: str) -> Path:
    return output_dir / f"{task}__{attack}__svdd.json"


def _complete(path: Path, task: str, attack: str, rounds: int, seed: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    meta = payload.get("meta", {})
    effective = meta.get("effective_config", {})
    records = payload.get("rounds")
    try:
        effective_seed = int(effective.get("seed", -1))
        total_rounds = int(meta.get("total_rounds", -1))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(records, list)
        and len(records) == rounds
        and isinstance(meta, dict)
        and meta.get("task") == task
        and meta.get("attack") == attack
        and meta.get("defense") == "svdd"
        and total_rounds == rounds
        and effective_seed == seed
    )


def _write_config(
    root: Path,
    task: str,
    attack: str,
    seed: int,
    rounds: int,
    factor: str,
    label: str,
    factor_overrides: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = root / task / attack / label / f"seed_{seed}"
    config_path = root / "_configs" / task / attack / label / f"seed_{seed}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update(factor_overrides)
    overrides.update({"seed": seed, "total_rounds": rounds})
    payload = {
        "task": task,
        "attacks": attack,
        "defenses": "svdd",
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path, output_dir


def _parse_csv(value: str, cast: type[str] | type[int]) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("log/svdd_cross_task_sensitivity_absolute_mad_topk10_40_300"))
    parser.add_argument("--tasks", default=",".join(ALL_TASKS))
    parser.add_argument("--attacks", default="gn,sf,lf,bd,lie")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tasks = tuple(item.lower() for item in _parse_csv(args.tasks, str))
    requested_attacks = tuple(item.lower() for item in _parse_csv(args.attacks, str))
    seeds = tuple(_parse_csv(args.seeds, int))
    gpus = tuple(_parse_csv(args.gpus, int))
    if not tasks or any(task not in ALL_TASKS for task in tasks):
        parser.error(f"tasks must be a subset of {ALL_TASKS}")
    if not requested_attacks or any(attack not in IMAGE_ATTACKS for attack in requested_attacks):
        parser.error(f"attacks must be a subset of {IMAGE_ATTACKS}")
    if not seeds or not gpus or args.rounds < 1 or args.workers_per_gpu < 1 or args.omp_threads < 1:
        parser.error("seeds, gpus, rounds, workers-per-gpu, and omp-threads must be positive")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = _factor_specs()
    jobs: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for task in tasks:
        allowed = set(IMAGE_ATTACKS if task in IMAGE_TASKS else AG_NEWS_ATTACKS)
        for attack in requested_attacks:
            if attack not in allowed:
                continue
            for factor, label, factor_overrides in specs:
                for seed in seeds:
                    entry: dict[str, Any] = {
                        "task": task,
                        "attack": attack,
                        "seed": seed,
                        "factor": factor,
                        "label": label,
                    }
                    # With N=100, the paper ALIE/LIE quantile is undefined at
                    # M >= 50. Keep these cells visible but do not fabricate a z.
                    if attack == "lie" and factor == "malicious_ratio" and factor_overrides["num_malicious"] >= 50:
                        entry["status"] = "invalid"
                        entry["reason"] = "paper LIE quantile undefined for malicious ratio >= 0.5"
                        invalid.append(entry)
                        continue
                    config_path, output_dir = _write_config(
                        root, task, attack, seed, args.rounds, factor, label, factor_overrides
                    )
                    result_path = _result_path(output_dir, task, attack)
                    entry.update({"status": "pending", "config": str(config_path), "result": str(result_path)})
                    if not args.force and _complete(result_path, task, attack, args.rounds, seed):
                        entry["status"] = "complete"
                    else:
                        jobs.append({**entry, "config_path": config_path, "output_dir": output_dir})

    manifest = {
        "description": "SVDD additive sensitivity matrix",
        "tasks": tasks,
        "attacks": requested_attacks,
        "seeds": seeds,
        "rounds": args.rounds,
        "svdd_input_mode": "absolute",
        "svdd_input_dim": 4096,
        "svdd_normalization": "median_mad",
        "topk_reject_ratios": [0.10, 0.20, 0.30, 0.40],
        "baseline": {"malicious_ratio": 0.3, "phase1_rounds": 15, "trust_size": 50, "latent_dim": 64},
        "invalid_jobs": invalid,
        "jobs": [{key: value for key, value in item.items() if key not in {"config_path", "output_dir"}} for item in jobs],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    expected_valid = sum(1 for task in tasks for attack in requested_attacks if attack in (IMAGE_ATTACKS if task in IMAGE_TASKS else AG_NEWS_ATTACKS)) * len(specs) * len(seeds) - len(invalid)
    completed = sum(1 for task in tasks for attack in requested_attacks if attack in (IMAGE_ATTACKS if task in IMAGE_TASKS else AG_NEWS_ATTACKS) for _factor, _label, _overrides in specs for _seed in seeds) - len(jobs) - len(invalid)
    print(
        f"pending={len(jobs)} complete={completed} invalid={len(invalid)} expected_valid={expected_valid} "
        f"rounds={args.rounds} gpus={gpus} workers_per_gpu={args.workers_per_gpu}",
        flush=True,
    )
    if not jobs:
        return 0

    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[dict[str, Any], int]] = []
    lock = threading.Lock()

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / "console.log"
            env = os.environ.copy()
            env.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": str(args.omp_threads),
                "MKL_NUM_THREADS": str(args.omp_threads),
                "OPENBLAS_NUM_THREADS": str(args.omp_threads),
                "PYTHONUNBUFFERED": "1",
            })
            command = [str(args.python_bin), "-u", "-m", "src.pipeline", "--config", str(job["config_path"])]
            with lock:
                print(
                    f"START gpu={gpu} worker={worker_id} task={job['task']} attack={job['attack']} "
                    f"factor={job['factor']}:{job['label']} seed={job['seed']}",
                    flush=True,
                )
            with console_path.open("w", encoding="utf-8") as stream:
                completed_process = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    start_new_session=True,
                )
            with lock:
                if completed_process.returncode == 0:
                    print(
                        f"DONE task={job['task']} attack={job['attack']} factor={job['label']} seed={job['seed']}",
                        flush=True,
                    )
                else:
                    failures.append((job, completed_process.returncode))
                    print(
                        f"FAIL task={job['task']} attack={job['attack']} factor={job['label']} "
                        f"seed={job['seed']} exit={completed_process.returncode} log={console_path}",
                        flush=True,
                    )
            pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, worker_id), daemon=False)
        for gpu in gpus
        for worker_id in range(args.workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print(f"failures={len(failures)}", flush=True)
        return 1
    print("all pending jobs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
