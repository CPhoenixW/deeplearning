#!/usr/bin/env python3
"""Supervise and retry the resumable RQ3 mechanism matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pid_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not workers or any(item < 1 for item in workers):
        raise ValueError("worker sequence must contain positive integers")
    return workers


def _runner_command(args: argparse.Namespace, workers: int) -> list[str]:
    return [
        str(args.python_bin),
        "tools/run_rq3_mechanism_matrix.py",
        "--task",
        args.task,
        "--mechanisms",
        args.mechanisms,
        "--attacks",
        args.attacks,
        "--malicious-ratios",
        args.malicious_ratios,
        "--seeds",
        args.seeds,
        "--rounds",
        str(args.rounds),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(workers),
        "--output-root",
        str(args.output_root),
        "--python",
        str(args.python_bin),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--worker-sequence", default="12,8,4,2,1")
    parser.add_argument("--task", default="fashion_mnist")
    parser.add_argument("--mechanisms", default="fedavg,p1_only,p2_only,full")
    parser.add_argument("--attacks", default="gn,lie,minmax,minsum,bd,mix")
    parser.add_argument("--malicious-ratios", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--gpus", default="0,2")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("log/rq3_mechanism_fashion_mnist_20260811"),
    )
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    try:
        worker_sequence = _parse_workers(args.worker_sequence)
    except ValueError as exc:
        parser.error(str(exc))

    if args.initial_pid is not None:
        print(f"WAIT initial_pid={args.initial_pid}", flush=True)
        while _pid_running(args.initial_pid):
            time.sleep(args.poll_seconds)
        print(f"INITIAL_DONE initial_pid={args.initial_pid}", flush=True)

    for pass_index, workers in enumerate(worker_sequence, start=1):
        command = _runner_command(args, workers)
        print(
            f"PASS_START index={pass_index} workers_per_gpu={workers} command={' '.join(command)}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        print(
            f"PASS_DONE index={pass_index} workers_per_gpu={workers} exit={completed.returncode}",
            flush=True,
        )
        if completed.returncode == 0:
            print("RQ3_COMPLETE", flush=True)
            return 0
    print("RQ3_INCOMPLETE retries_exhausted", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
