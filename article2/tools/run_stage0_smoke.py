#!/usr/bin/env python3
"""Run a small cross-task Stage 0 smoke matrix on available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFENSES = ("avg", "tm", "mk", "lasa", "seca", "bnguard", "dmc", "svdd")
CASES = (("mnist", "none"), ("fashion_mnist", "gn"), ("cifar10", "mix"), ("ag_news", "lie"))


def _write_config(root: Path, task: str, attack: str, defense: str, rounds: int) -> Path:
    output_dir = root / task / attack / defense
    path = root / "_configs" / task / attack / f"{defense}.json"
    overrides = {
        "num_clients": 20,
        "num_malicious": 6,
        "total_rounds": int(rounds),
        "server_validation_size": 20,
        "local_epochs": 1,
        "batch_size": 64,
        "num_workers": 0,
        "dirichlet_alpha": 1.0,
        "phase1_rounds": max(1, min(3, int(rounds) - 1)),
        "phase1_score_mode": "recon",
        "phase2_score_mode": "svdd",
        "alpha": 0.5,
        "mixed_attack_types": "lf,bd,gn,sf,lie,minmax,minsum",
        "param_descriptor_dim": 4096,
        "param_descriptor_device": "cuda",
        "device": "cuda",
    }
    payload = {
        "task": task,
        "attacks": attack,
        "defenses": defense,
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_jobs(root: Path, rounds: int) -> list[tuple[str, str, str, Path, Path]]:
    jobs = []
    for task, attack in CASES:
        for defense in DEFENSES:
            config = _write_config(root, task, attack, defense, rounds)
            jobs.append((task, attack, defense, config, root / task / attack / defense))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("log/article2_stage0"))
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 10 or args.rounds > 20:
        parser.error("Stage 0 rounds must be between 10 and 20")
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus or args.workers_per_gpu < 1:
        parser.error("at least one GPU and one worker per GPU are required")
    jobs = build_jobs(args.output_root.resolve(), args.rounds)
    if args.max_jobs is not None:
        jobs = jobs[: max(0, int(args.max_jobs))]
    print(f"jobs={len(jobs)} cases={list(CASES)} defenses={list(DEFENSES)} rounds={args.rounds}")
    for task, attack, defense, config, _output in jobs:
        print(f"PENDING task={task} attack={attack} defense={defense} config={config}")
    if args.dry_run or not jobs:
        return 0

    pending: queue.Queue[tuple[str, str, str, Path, Path]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                task, attack, defense, config, output_dir = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console = output_dir / "stage0.log"
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
            command = [str(args.python_bin), "-u", "-m", "src.pipeline", "--config", str(config)]
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
                    print(f"DONE task={task} attack={attack} defense={defense}", flush=True)
                else:
                    failures.append((task, attack, defense, int(completed.returncode)))
                    print(f"FAIL task={task} attack={attack} defense={defense} exit={completed.returncode}", flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus for _ in range(args.workers_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failures:")
        for task, attack, defense, code in failures:
            print(f"  {task}/{attack}/{defense}: exit={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
