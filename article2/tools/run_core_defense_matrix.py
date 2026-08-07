#!/usr/bin/env python3
"""Run the JSON-driven main-comparison defense matrix on one or more GPUs.

The experiment protocol defines FedAvg, Trimmed Mean, and Multi-Krum as the
primary comparison methods.  This scheduler gives every task × attack ×
defense combination its own JSON configuration and result directory, so jobs
can run concurrently while remaining independently resumable.
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
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DEFENSES = ("avg", "tm", "mk")
DEFAULT_ATTACKS = ("none", "gn", "lf", "sf", "bd", "lie", "mix")

# Match the C002-derived Fashion-MNIST AE-SVDD runs except for defense-specific
# parameters.  In particular, all methods use the calibrated task optimizer,
# population, partition, local training, attacks, and runtime settings.
BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
    "client_lr": 0.1,
    "client_momentum": 0.9,
    "client_weight_decay": 0.0,
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
    "hf_datasets_offline": True,
    "mixed_attack_types": "lf,bd,gn",
    "device": "cuda",
}


def _parse_csv(value: str, cast: type = str) -> tuple[Any, ...]:
    parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("CSV argument must not be empty.")
    return parsed


def _result_path(output_dir: Path, task: str, attack: str, defense: str) -> Path:
    return output_dir / f"{task}__{attack}__{defense}.json"


def _complete(
    output_dir: Path,
    *,
    task: str,
    attack: str,
    defense: str,
    rounds: int,
) -> bool:
    """Return whether the independently written result satisfies this job."""

    path = _result_path(output_dir, task, attack, defense)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    records = payload.get("rounds")
    meta = payload.get("meta")
    if not isinstance(records, list) or len(records) != int(rounds):
        return False
    if not isinstance(meta, dict):
        return False
    return (
        str(meta.get("task", "")) == task
        and str(meta.get("attack", "")) == attack
        and str(meta.get("defense", "")) == defense
        and int(meta.get("total_rounds", -1)) == int(rounds)
    )


def _write_config(
    root: Path,
    *,
    task: str,
    defense: str,
    attack: str,
    seed: int,
    rounds: int,
) -> tuple[Path, Path]:
    output_dir = root / defense / f"seed_{seed}"
    config_path = root / "_configs" / defense / f"seed_{seed}" / f"{attack}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update({"seed": int(seed), "total_rounds": int(rounds)})
    payload = {
        "task": task,
        "attacks": attack,
        "defenses": defense,
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config_path, output_dir


def _build_jobs(
    root: Path,
    *,
    task: str,
    defenses: Sequence[str],
    attacks: Sequence[str],
    seed: int,
    rounds: int,
    force: bool,
) -> list[tuple[str, str, Path, Path]]:
    jobs: list[tuple[str, str, Path, Path]] = []
    # Interleave defenses so a worker queue distributes computationally
    # different aggregators across all GPUs from the first wave.
    for attack in attacks:
        for defense in defenses:
            config_path, output_dir = _write_config(
                root,
                task=task,
                defense=defense,
                attack=attack,
                seed=seed,
                rounds=rounds,
            )
            if force or not _complete(
                output_dir,
                task=task,
                attack=attack,
                defense=defense,
                rounds=rounds,
            ):
                jobs.append((defense, attack, config_path, output_dir))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="fashion_mnist")
    parser.add_argument("--defenses", default=",".join(CORE_DEFENSES))
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("log/fashion_mnist_core_baselines"),
    )
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    defenses = _parse_csv(args.defenses)
    attacks = _parse_csv(args.attacks)
    gpus = _parse_csv(args.gpus, cast=int)
    unknown_defenses = sorted(set(defenses) - set(CORE_DEFENSES))
    unknown_attacks = sorted(set(attacks) - set(DEFAULT_ATTACKS))
    if unknown_defenses:
        parser.error(f"Only core comparison defenses are supported: {unknown_defenses}")
    if unknown_attacks:
        parser.error(f"Unsupported attack IDs: {unknown_attacks}")
    if args.rounds < 1 or args.workers_per_gpu < 1:
        parser.error("--rounds and --workers-per-gpu must be positive")

    root = args.output_root.resolve()
    jobs = _build_jobs(
        root,
        task=str(args.task),
        defenses=defenses,
        attacks=attacks,
        seed=int(args.seed),
        rounds=int(args.rounds),
        force=bool(args.force),
    )
    print(
        f"jobs={len(jobs)} task={args.task} defenses={list(defenses)} "
        f"attacks={list(attacks)} gpus={list(gpus)} "
        f"workers_per_gpu={args.workers_per_gpu}"
    )
    for defense, attack, config_path, _output_dir in jobs:
        print(f"PENDING defense={defense} attack={attack} config={config_path}")
    if args.dry_run or not jobs:
        return 0

    pending: queue.Queue[tuple[str, str, Path, Path]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, int, Path]] = []
    lock = threading.Lock()
    python_bin = str(args.python_bin)

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                defense, attack, config_path, output_dir = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / f"{attack}.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [python_bin, "-u", "-m", "src.pipeline", "--config", str(config_path)]
            with lock:
                print(
                    f"START gpu={gpu} worker={worker_id} defense={defense} "
                    f"attack={attack}",
                    flush=True,
                )
            with console_path.open("a", encoding="utf-8") as console:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                    start_new_session=True,
                )
            with lock:
                if completed.returncode == 0:
                    print(f"DONE  defense={defense} attack={attack}", flush=True)
                else:
                    failures.append((defense, attack, int(completed.returncode), console_path))
                    print(
                        f"FAIL  defense={defense} attack={attack} "
                        f"exit={completed.returncode} log={console_path}",
                        flush=True,
                    )
            pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, worker_id), daemon=False)
        for gpu in gpus
        for worker_id in range(int(args.workers_per_gpu))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failures:")
        for defense, attack, code, console_path in failures:
            print(f"  {defense}/{attack}: exit={code} log={console_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
