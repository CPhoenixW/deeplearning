#!/usr/bin/env python3
"""Launch a monitored Fashion-MNIST defense comparison matrix.

This is the single entry point for the standard AE-SVDD comparison protocol:
it starts the read-only browser monitor and invokes the resumable matrix
scheduler with the same output root.  Run it under ``nohup`` on a remote host
when the terminal should be released, then expose ``--monitor-port`` through
an SSH local forward.

Example:
    nohup .venv/bin/python tools/launch_fashion_comparison.py \\
      --dirichlet-alpha 0.05 --workers-per-gpu 8 \\
      --output-root log/fashion_alpha005_comparison > log/alpha005.launch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = PROJECT_ROOT / "tools" / "run_core_defense_matrix.py"
MONITOR_SCRIPT = PROJECT_ROOT / "tools" / "experiment_monitor.py"

DEFAULT_DEFENSES = "svdd,mk,seca,dmc"
DEFAULT_ATTACKS = "none,gn,lf,sf,bd,minmax,minsum,lit,scaling"
DEFAULT_SEEDS = "42,43,44"


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _resolve_output_root(value: str | None, alpha: float) -> Path:
    if value is None:
        tag = f"{alpha:.4f}".replace(".", "")
        value = f"log/fashion_alpha{tag}_comparison"
    root = Path(value)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _scheduler_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [
        args.python_bin,
        str(MATRIX_SCRIPT),
        "--task",
        "fashion_mnist",
        "--defenses",
        args.defenses,
        "--attacks",
        args.attacks,
        "--seeds",
        args.seeds,
        "--rounds",
        str(args.rounds),
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--omp-threads",
        str(args.omp_threads),
        "--output-root",
        str(output_root),
        "--python",
        args.python_bin,
        "--project-root",
        str(PROJECT_ROOT),
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.05)
    parser.add_argument("--defenses", default=DEFAULT_DEFENSES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--monitor-host", default="127.0.0.1")
    parser.add_argument("--monitor-port", type=int, default=18090)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    args = parser.parse_args(argv)

    if args.dirichlet_alpha <= 0.0:
        parser.error("--dirichlet-alpha must be positive")
    if args.rounds < 1 or args.workers_per_gpu < 1 or args.omp_threads < 1:
        parser.error("--rounds, --workers-per-gpu, and --omp-threads must be positive")

    output_root = _resolve_output_root(args.output_root, args.dirichlet_alpha)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.no_monitor:
        if _port_is_open(args.monitor_host, args.monitor_port):
            parser.error(
                f"monitor port {args.monitor_host}:{args.monitor_port} is already in use; "
                "choose --monitor-port or stop the existing monitor"
            )
        monitor_log = output_root / "monitor.log"
        with monitor_log.open("ab") as stream:
            subprocess.Popen(
                [
                    args.python_bin,
                    str(MONITOR_SCRIPT),
                    "--root",
                    str(output_root),
                    "--host",
                    args.monitor_host,
                    "--port",
                    str(args.monitor_port),
                ],
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            f"monitor=http://{args.monitor_host}:{args.monitor_port} "
            f"root={output_root}",
            flush=True,
        )

    command = _scheduler_command(args, output_root)
    print("scheduler=" + " ".join(command), flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
