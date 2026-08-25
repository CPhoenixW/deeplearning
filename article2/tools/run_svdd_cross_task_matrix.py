#!/usr/bin/env python3
"""Run the cross-task absolute-parameter AE-SVDD attack matrix."""

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


def _result_path(output_dir: Path, task: str, attack: str) -> Path:
    return output_dir / f"{task}__{attack}__svdd.json"


def _complete(path: Path, task: str, attack: str, rounds: int, seed: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    meta = payload.get("meta", {})
    records = payload.get("rounds")
    effective = meta.get("effective_config", {})
    return (
        isinstance(records, list)
        and len(records) == rounds
        and isinstance(meta, dict)
        and meta.get("task") == task
        and meta.get("attack") == attack
        and meta.get("defense") == "svdd"
        and int(meta.get("total_rounds", -1)) == rounds
        and int(effective.get("seed", -1)) == seed
    )


def _write_config(root: Path, task: str, attack: str, seed: int, rounds: int) -> tuple[Path, Path]:
    output_dir = root / task / attack / f"seed_{seed}"
    config_path = root / "_configs" / task / attack / f"seed_{seed}.json"
    overrides = dict(BASE_OVERRIDES)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("log/svdd_cross_task_absolute_mad_topk10_40_300"))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    if not seeds or not gpus or args.rounds < 1 or args.workers_per_gpu < 1 or args.omp_threads < 1:
        parser.error("seeds, gpus, rounds, workers-per-gpu, and omp-threads must be positive")

    root = args.output_root.resolve()
    jobs: list[tuple[str, str, int, Path, Path]] = []
    for task in ALL_TASKS:
        attacks = IMAGE_ATTACKS if task in IMAGE_TASKS else AG_NEWS_ATTACKS
        for attack in attacks:
            for seed in seeds:
                config_path, output_dir = _write_config(root, task, attack, seed, args.rounds)
                if args.force or not _complete(
                    _result_path(output_dir, task, attack), task, attack, args.rounds, seed
                ):
                    jobs.append((task, attack, seed, config_path, output_dir))

    expected = (len(IMAGE_TASKS) * len(IMAGE_ATTACKS) + len(AG_NEWS_ATTACKS)) * len(seeds)
    print(
        f"jobs={len(jobs)} expected={expected} rounds={args.rounds} "
        f"gpus={gpus} workers_per_gpu={args.workers_per_gpu} "
        f"image_attacks={IMAGE_ATTACKS} ag_news_attacks={AG_NEWS_ATTACKS}",
        flush=True,
    )
    if not jobs:
        return 0

    pending: queue.Queue[tuple[str, str, int, Path, Path]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, int, Path]] = []
    lock = threading.Lock()

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                task, attack, seed, config_path, output_dir = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / "console.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": str(args.omp_threads),
                    "MKL_NUM_THREADS": str(args.omp_threads),
                    "OPENBLAS_NUM_THREADS": str(args.omp_threads),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [str(args.python_bin), "-u", "-m", "src.pipeline", "--config", str(config_path)]
            with lock:
                print(f"START gpu={gpu} worker={worker_id} task={task} attack={attack} seed={seed}", flush=True)
            with console_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    start_new_session=True,
                )
            with lock:
                if completed.returncode == 0:
                    print(f"DONE task={task} attack={attack} seed={seed}", flush=True)
                else:
                    failures.append((task, attack, seed, console_path))
                    print(f"FAIL task={task} attack={attack} seed={seed} exit={completed.returncode}", flush=True)
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
        print("failures:")
        for task, attack, seed, path in failures:
            print(f"  {task}/{attack}/seed_{seed}: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
