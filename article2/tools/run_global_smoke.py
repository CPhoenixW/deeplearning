#!/usr/bin/env python3
"""Run a resumable all-task, all-attack, all-defense preflight matrix."""

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
ATTACKS = ("none", "lf", "gn", "sf", "bd", "lie", "minmax", "minsum")
DEFENSES = (
    "avg",
    "tm",
    "mk",
    "lasa",
    "seca",
    "bnguard",
    "dmc",
    "svdd",
    "fld",
    "alignins",
    "flgmm",
    "flanders",
)


def _overrides(rounds: int) -> dict[str, Any]:
    return {
        "num_clients": 5,
        "num_malicious": 2,
        "total_rounds": int(rounds),
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
        "phase1_rounds": 3,
        "svdd_lambda": 0.5,
        "svdd_normalization_eps": 1e-6,
        "device": "cuda",
    }


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
    path = _result_path(output_dir, task, attack, defense)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        records = payload["rounds"]
        effective = meta["effective_config"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected_malicious = 0 if attack == "none" else 2
    return (
        meta.get("task") == task
        and meta.get("attack") == attack
        and meta.get("defense") == defense
        and int(meta.get("total_rounds", -1)) == int(rounds)
        and len(records) == int(rounds)
        and int(effective.get("num_clients", -1)) == 5
        and int(effective.get("num_clients", -1))
        - int(effective.get("num_benign", -1))
        == expected_malicious
        and int(effective.get("local_epochs", -1)) == 1
        and int(effective.get("phase1_rounds", -1)) == 3
    )


def _write_config(
    root: Path,
    *,
    task: str,
    attack: str,
    defense: str,
    rounds: int,
) -> tuple[Path, Path]:
    output_dir = (root / task / attack / defense).resolve()
    config_path = (root / "_configs" / task / attack / f"{defense}.json").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "attacks": attack,
        "defenses": defense,
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": _overrides(rounds),
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path, output_dir


def _jobs(root: Path, rounds: int) -> list[tuple[str, str, str, Path, Path]]:
    jobs = []
    for task in TASKS:
        for attack in ATTACKS:
            for defense in DEFENSES:
                config, output_dir = _write_config(
                    root,
                    task=task,
                    attack=attack,
                    defense=defense,
                    rounds=rounds,
                )
                jobs.append((task, attack, defense, config, output_dir))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path, default=Path("log/global_smoke_5c_5r")
    )
    parser.add_argument("--python", dest="python_bin", default="dl/bin/python")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be at least 1")
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be at least 1")
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("at least one GPU is required")
    python_bin = Path(args.python_bin)
    if not python_bin.is_absolute():
        python_bin = (PROJECT_ROOT / python_bin).resolve()
    if not python_bin.exists():
        parser.error(f"Python executable does not exist: {python_bin}")

    root = args.output_root
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    jobs = _jobs(root, args.rounds)
    pending_jobs = [
        job
        for job in jobs
        if not _complete(
            job[4], task=job[0], attack=job[1], defense=job[2], rounds=args.rounds
        )
    ]
    if args.max_jobs is not None:
        pending_jobs = pending_jobs[: max(0, int(args.max_jobs))]

    print(
        f"jobs={len(jobs)} pending={len(pending_jobs)} tasks={len(TASKS)} "
        f"attacks={len(ATTACKS)} defenses={len(DEFENSES)} rounds={args.rounds}",
        flush=True,
    )
    if args.dry_run or not pending_jobs:
        return 0

    pending: queue.Queue[tuple[str, str, str, Path, Path]] = queue.Queue()
    for job in pending_jobs:
        pending.put(job)
    failures: list[tuple[str, str, str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                task, attack, defense, config, output_dir = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console = output_dir / "smoke.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [str(python_bin), "-u", "-m", "src.pipeline", "--config", str(config)]
            with console.open("a", encoding="utf-8") as stream:
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
                    print(
                        f"DONE gpu={gpu}/w{worker_id} task={task} "
                        f"attack={attack} defense={defense}",
                        flush=True,
                    )
                else:
                    failures.append((task, attack, defense, int(completed.returncode)))
                    print(
                        f"FAIL gpu={gpu}/w{worker_id} task={task} "
                        f"attack={attack} defense={defense} exit={completed.returncode}",
                        flush=True,
                    )
            pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, index), daemon=False)
        for gpu in gpus
        for index in range(args.workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failures:", flush=True)
        for task, attack, defense, code in failures:
            print(f"  {task}/{attack}/{defense}: exit={code}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
