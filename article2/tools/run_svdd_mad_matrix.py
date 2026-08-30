#!/usr/bin/env python3
"""Run the Fashion-MNIST SVDD median-plus-k-MAD selection matrix."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ("gn", "lf", "bd", "sf")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_KS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

BASE_OVERRIDES = {
    "num_clients": 100,
    "num_malicious": 30,
    "server_validation_size": 200,
    "latent_dim": 64,
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
    "phase2_score_mode": "robust_z",
    "svdd_selection_method": "mad_threshold",
    "svdd_validation_tie_break": "median",
    "svdd_lambda": 0.5,
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "device": "cuda",
}


def _parse_csv(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _write_config(root: Path, k: float, attack: str, seed: int, rounds: int):
    k_label = f"k_{_label(k)}"
    output_dir = root / k_label / attack / f"seed_{seed}"
    config_path = root / "_configs" / k_label / attack / f"seed_{seed}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update({
        "seed": seed,
        "total_rounds": rounds,
        "svdd_mad_k": float(k),
    })
    payload = {
        "task": "fashion_mnist",
        "attacks": attack,
        "defenses": "svdd",
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return config_path, output_dir, output_dir / f"fashion_mnist__{attack}__svdd.json"


def _complete(path: Path, attack: str, seed: int, rounds: int) -> bool:
    try:
        payload = json.loads(path.read_text())
        meta = payload.get("meta", {})
        cfg = meta.get("effective_config", {})
        return (
            isinstance(payload.get("rounds"), list)
            and len(payload["rounds"]) == rounds
            and meta.get("task") == "fashion_mnist"
            and meta.get("attack") == attack
            and int(cfg.get("seed", -1)) == seed
            and cfg.get("svdd_selection_method") == "mad_threshold"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    seeds = _parse_csv(args.seeds, int)
    ks = _parse_csv(args.k_values, float)
    gpus = _parse_csv(args.gpus, int)
    if not seeds or not ks or not gpus or args.rounds < 1 or args.workers_per_gpu < 1:
        parser.error("seeds, k-values, gpus, rounds, and workers-per-gpu must be positive")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in ks:
        for attack in ATTACKS:
            for seed in seeds:
                config, output, result = _write_config(root, k, attack, seed, args.rounds)
                jobs.append({"k": k, "attack": attack, "seed": seed, "config": config,
                             "output": output, "result": result})

    pending = []
    complete = 0
    for job in jobs:
        if not args.force and _complete(job["result"], job["attack"], job["seed"], args.rounds):
            complete += 1
        else:
            pending.append(job)
    manifest = {
        "description": "Fashion-MNIST SVDD median+k*MAD threshold sensitivity",
        "task": "fashion_mnist", "attacks": ATTACKS, "seeds": seeds,
        "k_values": ks, "rounds": args.rounds, "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu, "base_overrides": BASE_OVERRIDES,
        "requested_jobs": len(jobs), "complete_before_run": complete,
        "pending_before_run": len(pending),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"requested={len(jobs)} complete={complete} pending={len(pending)}", flush=True)
    if args.prepare_only or not pending:
        return 0

    queues = {gpu: queue.Queue() for gpu in gpus}
    for index, job in enumerate(pending):
        queues[gpus[index % len(gpus)]].put(job)
    print("gpu_pending=" + ",".join(f"{gpu}:{queues[gpu].qsize()}" for gpu in gpus), flush=True)
    lock = threading.Lock()
    failures = []

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                job = queues[gpu].get_nowait()
            except queue.Empty:
                return
            job["output"].mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": str(args.omp_threads),
                "MKL_NUM_THREADS": str(args.omp_threads),
                "OPENBLAS_NUM_THREADS": str(args.omp_threads),
                "PYTHONUNBUFFERED": "1",
            })
            command = [str(args.python_bin), "-u", "-m", "src.pipeline", "--config", str(job["config"])]
            console = job["output"] / "console.log"
            with lock:
                print(f"START gpu={gpu} worker={worker_id} k={job['k']} {job['attack']}/seed_{job['seed']}", flush=True)
            with console.open("w", encoding="utf-8") as stream:
                completed_process = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env,
                                                   stdout=stream, stderr=subprocess.STDOUT,
                                                   check=False, start_new_session=True)
            with lock:
                if completed_process.returncode == 0:
                    print(f"DONE k={job['k']} {job['attack']}/seed_{job['seed']}", flush=True)
                else:
                    failures.append((job, completed_process.returncode))
                    print(f"FAIL k={job['k']} {job['attack']}/seed_{job['seed']} exit={completed_process.returncode}", flush=True)
            queues[gpu].task_done()

    threads = []
    for gpu in gpus:
        for worker_id in range(args.workers_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu, worker_id), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    print(f"finished failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
