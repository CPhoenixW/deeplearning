#!/usr/bin/env python3
"""Run the 5.2 one-factor AE-SVDD sensitivity matrix.

The matrix is intentionally additive, not a Cartesian product: seven
``svdd_lambda`` values, four Phase-1 lengths, and four latent dimensions are
run independently for each of the five datasets. Every job uses GN, SVDD,
300 rounds, and one seed.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("mnist", "fashion_mnist", "cifar10", "covid19", "ag_news")
DEFAULT_LAMBDAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_PHASE1_ROUNDS = (5, 10, 15, 30, 50, 100)
DEFAULT_VALIDATION_SIZES = (10, 50, 100, 500)
DEFAULT_LATENT_DIMS = (8, 32, 64, 256, 512, 4096)
DEFAULT_TASKS = ("mnist", "fashion_mnist")


def _parse_csv(value: str, cast: type) -> tuple[Any, ...]:
    parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("CSV argument must not be empty")
    return parsed


def _format_value(value: float | int) -> str:
    return str(value).replace(".", "p")


def _factor_specs(
    lambdas: tuple[float, ...],
    phase1_rounds: tuple[int, ...],
    validation_sizes: tuple[int, ...],
    latent_dims: tuple[int, ...],
) -> list[tuple[str, float, int, int, int]]:
    specs: list[tuple[str, float, int, int, int]] = []
    for value in lambdas:
        specs.append((f"lambda_{_format_value(value)}", value, 15, 64, 50))
    for value in phase1_rounds:
        specs.append((f"phase1_{int(value):03d}", 0.5, int(value), 64, 50))
    for value in validation_sizes:
        specs.append((f"validation_{int(value):03d}", 0.5, 15, 64, int(value)))
    for value in latent_dims:
        specs.append((f"latent_{int(value):04d}", 0.5, 15, int(value), 50))
    return specs


def _overrides(
    *,
    svdd_lambda: float,
    phase1_rounds: int,
    latent_dim: int,
    validation_size: int,
    seed: int,
    rounds: int,
) -> dict[str, Any]:
    return {
        "num_clients": 100,
        "num_malicious": 30,
        "total_rounds": int(rounds),
        "server_validation_size": int(validation_size),
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
        "phase1_rounds": int(phase1_rounds),
        "svdd_lambda": float(svdd_lambda),
        "latent_dim": int(latent_dim),
        "svdd_normalization_eps": 1e-6,
        "device": "cuda",
        "seed": int(seed),
    }


def _write_config(
    root: Path,
    *,
    task: str,
    factor: str,
    svdd_lambda: float,
    phase1_rounds: int,
    latent_dim: int,
    validation_size: int,
    seed: int,
    rounds: int,
) -> tuple[Path, Path]:
    output_dir = (root / task / factor / f"seed_{seed}").resolve()
    config_path = (root / "_configs" / task / factor / f"seed_{seed}.json").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "attacks": "gn",
        "defenses": "svdd",
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": _overrides(
            svdd_lambda=svdd_lambda,
            phase1_rounds=phase1_rounds,
            latent_dim=latent_dim,
            validation_size=validation_size,
            seed=seed,
            rounds=rounds,
        ),
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path, output_dir


def _complete(
    output_dir: Path,
    *,
    task: str,
    factor: str,
    svdd_lambda: float,
    phase1_rounds: int,
    latent_dim: int,
    validation_size: int,
    seed: int,
    rounds: int,
) -> bool:
    path = output_dir / f"{task}__gn__svdd.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        effective = meta["effective_config"]
        records = payload["rounds"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        meta.get("task") == task
        and meta.get("attack") == "gn"
        and meta.get("defense") == "svdd"
        and int(meta.get("total_rounds", -1)) == int(rounds)
        and len(records) == int(rounds)
        and int(effective.get("seed", -1)) == int(seed)
        and int(effective.get("num_clients", -1)) == 100
        and int(effective.get("num_clients", -1)) - int(effective.get("num_benign", -1)) == 30
        and int(effective.get("phase1_rounds", -1)) == int(phase1_rounds)
        and int(effective.get("latent_dim", -1)) == int(latent_dim)
        and int(effective.get("server_validation_size", -1)) == int(validation_size)
        and abs(float(effective.get("svdd_lambda", -1.0)) - float(svdd_lambda)) < 1e-8
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--phase1-rounds", default=",".join(map(str, DEFAULT_PHASE1_ROUNDS)))
    parser.add_argument("--validation-sizes", default=",".join(map(str, DEFAULT_VALIDATION_SIZES)))
    parser.add_argument("--latent-dims", default=",".join(map(str, DEFAULT_LATENT_DIMS)))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--factors",
        default=None,
        help="Optional comma-separated factor names to run, e.g. phase1_100 or latent_0064,lambda_0p5.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("log/svdd_5_2_sensitivity"))
    parser.add_argument("--python", dest="python_bin", default="dl/bin/python")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lambdas = tuple(float(value) for value in _parse_csv(args.lambdas, float))
    phase1_rounds = tuple(int(value) for value in _parse_csv(args.phase1_rounds, int))
    validation_sizes = tuple(int(value) for value in _parse_csv(args.validation_sizes, int))
    latent_dims = tuple(int(value) for value in _parse_csv(args.latent_dims, int))
    tasks = tuple(str(value) for value in _parse_csv(args.tasks, str))
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        parser.error(f"unknown tasks: {unknown_tasks}; choose from {TASKS}")
    if any(not 0.0 <= value <= 1.0 for value in lambdas):
        parser.error("all svdd_lambda values must be in [0, 1]")
    if any(value < 1 or value >= args.rounds for value in phase1_rounds):
        parser.error("phase1 rounds must be positive and less than total rounds")
    if any(value < 1 for value in validation_sizes):
        parser.error("validation sizes must be positive")
    if any(value < 1 for value in latent_dims):
        parser.error("latent dimensions must be positive")
    if args.rounds < 1 or args.workers_per_gpu < 1:
        parser.error("rounds and workers-per-gpu must be positive")
    gpus = tuple(int(value) for value in _parse_csv(args.gpus, int))
    python_bin = Path(args.python_bin)
    if not python_bin.is_absolute():
        python_bin = (PROJECT_ROOT / python_bin).resolve()
    if not python_bin.exists():
        parser.error(f"Python executable does not exist: {python_bin}")
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    root = root.resolve()

    specs = _factor_specs(lambdas, phase1_rounds, validation_sizes, latent_dims)
    if args.factors:
        requested_factors = tuple(str(value) for value in _parse_csv(args.factors, str))
        available_factors = {factor for factor, *_ in specs}
        unknown_factors = sorted(set(requested_factors) - available_factors)
        if unknown_factors:
            parser.error(f"unknown factors: {unknown_factors}; choose from {sorted(available_factors)}")
        selected = set(requested_factors)
        specs = [spec for spec in specs if spec[0] in selected]
    jobs: list[tuple[str, str, Path, Path]] = []
    for task in tasks:
        for factor, svdd_lambda, p1, latent_dim, validation_size in specs:
            config, output = _write_config(
                root,
                task=task,
                factor=factor,
                svdd_lambda=svdd_lambda,
                phase1_rounds=p1,
                latent_dim=latent_dim,
                validation_size=validation_size,
                seed=args.seed,
                rounds=args.rounds,
            )
            if not _complete(
                output,
                task=task,
                factor=factor,
                svdd_lambda=svdd_lambda,
                phase1_rounds=p1,
                latent_dim=latent_dim,
                validation_size=validation_size,
                seed=args.seed,
                rounds=args.rounds,
            ):
                jobs.append((task, factor, config, output))
    if args.max_jobs is not None:
        jobs = jobs[: max(0, int(args.max_jobs))]
    print(
        f"expected={len(tasks) * len(specs)} pending={len(jobs)} tasks={len(tasks)} "
        f"factors={len(specs)} rounds={args.rounds} seed={args.seed} attack=gn defense=svdd",
        flush=True,
    )
    if args.dry_run or not jobs:
        return 0

    worker_count = len(gpus) * args.workers_per_gpu
    worker_queues: list[queue.Queue[tuple[str, str, Path, Path]]] = [queue.Queue() for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        worker_queues[index % worker_count].put(job)
    failures: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int, worker_id: int, work_queue: queue.Queue[tuple[str, str, Path, Path]]) -> None:
        while True:
            try:
                task, factor, config, output = work_queue.get_nowait()
            except queue.Empty:
                return
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "sensitivity.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [str(python_bin), "-u", "-m", "src.pipeline", "--config", str(config)]
            with log_path.open("a", encoding="utf-8") as stream:
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
                    print(f"DONE gpu={gpu}/w{worker_id} task={task} factor={factor}", flush=True)
                else:
                    failures.append((task, factor, int(completed.returncode)))
                    print(f"FAIL gpu={gpu}/w{worker_id} task={task} factor={factor} exit={completed.returncode}", flush=True)
            work_queue.task_done()

    threads = []
    worker_index = 0
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            threads.append(
                threading.Thread(
                    target=worker,
                    args=(gpu, worker_index, worker_queues[worker_index]),
                    daemon=False,
                )
            )
            worker_index += 1
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failures:", flush=True)
        for task, factor, code in failures:
            print(f"  {task}/{factor}: exit={code}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
