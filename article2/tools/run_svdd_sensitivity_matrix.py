#!/usr/bin/env python3
"""Run a resumable AE-SVDD sensitivity matrix across multiple GPUs.

The matrix varies the Phase-1 duration, malicious-client fraction, and the
two-phase score/loss mode.  Every seed/attack/factor combination gets an
independent JSON configuration and output directory, so one numerical failure
does not discard the rest of the matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_ROUNDS = (5, 15, 30, 50)
DEFAULT_MALICIOUS_RATIOS = (0.10, 0.20, 0.30, 0.40)
DEFAULT_MODES = ("recon", "combined", "svdd")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ATTACKS = ("none", "gn", "lf", "sf", "bd", "lie", "mix")
DEFAULT_SVDD_LAMBDA = 0.5

# AG News rank-1 calibration and the C002-derived defense/runtime protocol.
BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "total_rounds": 100,
    "client_lr": 0.1,
    "client_momentum": 0.9,
    "client_weight_decay": 0.0005,
    "local_epochs": 1,
    "batch_size": 64,
    "num_workers": 0,
    "use_amp": False,
    "channels_last": False,
    "cuda_aggregation": True,
    "reuse_client_model": True,
    "skip_redundant_attack_training": True,
    "client_batch_group_size": 5,
    "round_diagnostics": False,
    "dirichlet_alpha": None,
    "hf_datasets_offline": True,
    "mixed_attack_types": "lf,bd,gn",
    "latent_dim": 64,
    "ae_lr": 0.001,
    "ae_weight_decay": 1e-6,
    "ae_grad_clip": 1.0,
    "svdd_input_mode": "delta",
    "svdd_input_dim": 4096,
    "svdd_normalization_eps": 1e-6,
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "device": "cuda",
}


def _parse_csv(value: str, cast: type = str) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("CSV argument must not be empty")
    return values


def _parse_ratios(value: str) -> tuple[float, ...]:
    ratios = _parse_csv(value, cast=float)
    for ratio in ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"malicious ratio must be in (0, 1), got {ratio}")
    return tuple(float(ratio) for ratio in ratios)


def _factor_id(phase1_rounds: int, malicious_ratio: float, mode: str, svdd_lambda: float) -> str:
    lambda_id = f"{svdd_lambda:g}".replace(".", "p")
    return f"p1_{int(phase1_rounds):03d}/mal_{int(round(100 * malicious_ratio)):02d}/{mode}/lambda_{lambda_id}"


def _result_path(output_dir: Path, task: str, attack: str) -> Path:
    return output_dir / f"{task}__{attack}__svdd.json"


def _complete(
    path: Path,
    *,
    task: str,
    attack: str,
    phase1_rounds: int,
    malicious_ratio: float,
    mode: str,
    svdd_lambda: float,
    seed: int,
    rounds: int,
) -> bool:
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
    effective = meta.get("effective_config", {})
    if not isinstance(effective, dict):
        return False
    expected_malicious = int(round(100 * malicious_ratio))
    return (
        str(meta.get("task", "")) == task
        and str(meta.get("attack", "")) == attack
        and str(meta.get("defense", "")) == "svdd"
        and int(meta.get("total_rounds", -1)) == int(rounds)
        and int(effective.get("phase1_rounds", -1)) == int(phase1_rounds)
        and str(effective.get("phase1_score_mode", "")) == "recon"
        and str(effective.get("phase2_score_mode", "")) == mode
        and abs(float(effective.get("svdd_lambda", -1.0)) - float(svdd_lambda)) < 1e-8
        and int(effective.get("seed", -1)) == int(seed)
        and (
            attack == "none"
            or int(effective.get("num_clients", -1)) - int(effective.get("num_benign", -1))
            == expected_malicious
        )
    )


def _write_config(
    root: Path,
    *,
    task: str,
    phase1_rounds: int,
    malicious_ratio: float,
    mode: str,
    svdd_lambda: float,
    seed: int,
    attack: str,
    rounds: int,
) -> tuple[Path, Path]:
    factor = _factor_id(phase1_rounds, malicious_ratio, mode, svdd_lambda)
    output_dir = root / factor / f"seed_{seed}"
    config_path = root / "_configs" / factor / f"seed_{seed}" / f"{attack}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update(
        {
            "seed": int(seed),
            "total_rounds": int(rounds),
            "phase1_rounds": int(phase1_rounds),
            "num_malicious": int(round(100 * malicious_ratio)),
            "phase1_score_mode": "recon",
            "phase2_score_mode": mode,
            "svdd_lambda": float(svdd_lambda),
        }
    )
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
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config_path, output_dir


def _build_jobs(
    root: Path,
    *,
    task: str,
    phase1_rounds: Sequence[int],
    malicious_ratios: Sequence[float],
    modes: Sequence[str],
    svdd_lambda: float,
    seeds: Sequence[int],
    attacks: Sequence[str],
    rounds: int,
    force: bool,
    max_jobs: int | None,
    clean_reference_ratio: float,
) -> list[tuple[str, str, Path, Path, int, float, str, int]]:
    jobs: list[tuple[str, str, Path, Path, int, float, str, int]] = []
    # Interleave factors and attacks so every GPU sees a mix of client counts,
    # phases, and score modes instead of receiving one expensive block.
    factors = [
        (int(p1), float(ratio), str(mode))
        for p1 in phase1_rounds
        for ratio in malicious_ratios
        for mode in modes
    ]
    # A clean run is independent of ``num_malicious``.  Keep one reference
    # ratio so the clean control does not multiply the matrix by four.
    has_clean_attack = "none" in attacks
    for seed in seeds:
        for p1, ratio, mode in factors:
            for attack in attacks:
                if (
                    attack == "none"
                    and has_clean_attack
                    and abs(float(ratio) - float(clean_reference_ratio)) > 1e-9
                ):
                    continue
                config_path, output_dir = _write_config(
                    root,
                    task=task,
                    phase1_rounds=p1,
                    malicious_ratio=ratio,
                    mode=mode,
                    svdd_lambda=svdd_lambda,
                    seed=int(seed),
                    attack=attack,
                    rounds=rounds,
                )
                if force or not _complete(
                    _result_path(output_dir, task, attack),
                    task=task,
                    attack=attack,
                    phase1_rounds=p1,
                    malicious_ratio=ratio,
                    mode=mode,
                    svdd_lambda=svdd_lambda,
                    seed=int(seed),
                    rounds=rounds,
                ):
                    jobs.append(
                        (
                            _factor_id(p1, ratio, mode, svdd_lambda),
                            attack,
                            config_path,
                            output_dir,
                            p1,
                            ratio,
                            mode,
                            int(seed),
                        )
                    )
                    if max_jobs is not None and len(jobs) >= int(max_jobs):
                        return jobs
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="fashion_mnist")
    parser.add_argument("--phase1-rounds", default=",".join(map(str, DEFAULT_PHASE1_ROUNDS)))
    parser.add_argument("--malicious-ratios", default=",".join(map(str, DEFAULT_MALICIOUS_RATIOS)))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--svdd-lambda", type=float, default=DEFAULT_SVDD_LAMBDA)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=12)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--time-limit-hours",
        type=float,
        default=None,
        help="Shared hard wall-clock limit; active child runs are killed at the deadline.",
    )
    parser.add_argument(
        "--clean-reference-ratio",
        type=float,
        default=0.30,
        help="Run the attack=none control only at this ratio (the ratio has no effect for clean clients).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("log/fashion_mnist_svdd_sensitivity_screen"))
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit queued jobs for a smoke test.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phase1_rounds = _parse_csv(args.phase1_rounds, cast=int)
    ratios = _parse_ratios(args.malicious_ratios)
    modes = tuple(item.strip().lower() for item in _parse_csv(args.modes))
    seeds = _parse_csv(args.seeds, cast=int)
    attacks = _parse_csv(args.attacks)
    gpus = _parse_csv(args.gpus, cast=int)
    unknown_modes = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown_modes:
        parser.error(f"Unsupported modes: {unknown_modes}")
    if min(phase1_rounds) < 1 or args.rounds < 1:
        parser.error("phase1 rounds and total rounds must be positive")
    if not 0.0 <= float(args.svdd_lambda) <= 1.0:
        parser.error("svdd-lambda must be in [0, 1]")
    if any(int(p1) >= int(args.rounds) for p1 in phase1_rounds):
        parser.error("each phase1 round count must be less than total rounds")
    if args.workers_per_gpu < 1 or args.omp_threads < 1:
        parser.error("workers-per-gpu and omp-threads must be positive")
    if args.time_limit_hours is not None and args.time_limit_hours <= 0.0:
        parser.error("time-limit-hours must be positive")
    if "none" in attacks and not any(
        abs(float(ratio) - float(args.clean_reference_ratio)) <= 1e-9
        for ratio in ratios
    ):
        parser.error("clean-reference-ratio must be one of --malicious-ratios when attack=none is selected")

    root = args.output_root.resolve()
    jobs = _build_jobs(
        root,
        task=str(args.task),
        phase1_rounds=phase1_rounds,
        malicious_ratios=ratios,
        modes=modes,
        svdd_lambda=float(args.svdd_lambda),
        seeds=seeds,
        attacks=attacks,
        rounds=int(args.rounds),
        force=bool(args.force),
        max_jobs=args.max_jobs,
        clean_reference_ratio=float(args.clean_reference_ratio),
    )
    clean_factor_count = 1 if "none" in attacks else 0
    non_clean_attack_count = len(attacks) - clean_factor_count
    total_expected = (
        len(phase1_rounds)
        * len(modes)
        * len(seeds)
        * (len(ratios) * non_clean_attack_count + clean_factor_count)
    )
    print(
        f"jobs={len(jobs)} expected={total_expected} task={args.task} rounds={args.rounds} "
        f"gpus={list(gpus)} workers_per_gpu={args.workers_per_gpu} omp_threads={args.omp_threads} "
        f"time_limit_hours={args.time_limit_hours}"
    )
    for factor, attack, config_path, _out, _p1, _ratio, _mode, seed in jobs[:80]:
        print(f"PENDING seed={seed} factor={factor} attack={attack} config={config_path}")
    if len(jobs) > 80:
        print(f"... {len(jobs) - 80} more jobs")
    if args.dry_run or not jobs:
        return 0

    pending: queue.Queue[tuple[str, str, Path, Path, int, float, str, int]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, int, Path]] = []
    lock = threading.Lock()
    python_bin = str(args.python_bin)
    deadline = (
        time.monotonic() + 3600.0 * float(args.time_limit_hours)
        if args.time_limit_hours is not None
        else None
    )
    deadline_reached = threading.Event()

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            if deadline_reached.is_set():
                return
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                deadline_reached.set()
                return
            try:
                factor, attack, config_path, output_dir, _p1, _ratio, _mode, seed = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / f"{attack}.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": str(int(args.omp_threads)),
                    "MKL_NUM_THREADS": str(int(args.omp_threads)),
                    "OPENBLAS_NUM_THREADS": str(int(args.omp_threads)),
                    "PYTHONUNBUFFERED": "1",
                    # These must be set before ``datasets`` is imported.  The
                    # task-level config switch is otherwise too late to stop
                    # one failed HuggingFace HEAD request per worker.
                    "HF_DATASETS_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            command = [python_bin, "-u", "-m", "src.pipeline", "--config", str(config_path)]
            with lock:
                print(f"START gpu={gpu} worker={worker_id} seed={seed} factor={factor} attack={attack}", flush=True)
            with console_path.open("a", encoding="utf-8") as console:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                        check=False,
                        start_new_session=True,
                        timeout=remaining,
                    )
                except subprocess.TimeoutExpired:
                    deadline_reached.set()
                    with lock:
                        failures.append((factor, attack, 124, console_path))
                        print(
                            f"TIMEOUT seed={seed} factor={factor} attack={attack} log={console_path}",
                            flush=True,
                        )
                    pending.task_done()
                    return
            with lock:
                if completed.returncode == 0:
                    print(f"DONE  seed={seed} factor={factor} attack={attack}", flush=True)
                else:
                    failures.append((factor, attack, int(completed.returncode), console_path))
                    print(f"FAIL  seed={seed} factor={factor} attack={attack} exit={completed.returncode} log={console_path}", flush=True)
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
        for factor, attack, code, log_path in failures:
            print(f"  {factor}/{attack}: exit={code} log={log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
