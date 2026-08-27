#!/usr/bin/env python3
"""Run the approved AE-SVDD sensitivity and malicious-ratio robustness matrix."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("mnist", "fashion_mnist", "cifar10", "ag_news")
IMAGE_TASKS = {"mnist", "fashion_mnist", "cifar10"}
IMAGE_ATTACKS = ("gn", "sf", "lf", "bd")
AG_NEWS_ATTACKS = ("gn", "sf", "lf")
SEEDS = (42, 43, 44)

# Conservative peak-memory estimates for the current 24-GB GPUs.  These are
# admission-control estimates, not changes to the experiment itself.
TASK_MEMORY_GB = {
    "mnist": 1.0,
    "fashion_mnist": 1.5,
    "cifar10": 5.5,
    "ag_news": 8.0,
}

BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
    "server_validation_size": 50,
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
    "svdd_normalization_eps": 1e-6,
    "svdd_descriptor_device": "cuda",
    "phase1_rounds": 15,
    "svdd_lambda": 0.5,
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "device": "cuda",
}


def _parameter_specs() -> list[tuple[str, str, dict[str, Any]]]:
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for value in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        specs.append(("lambda", f"lambda_{value:.1f}".replace(".", "p"), {"svdd_lambda": value}))
    for value in (5, 15, 30, 50, 100):
        specs.append(("phase1", f"phase1_{value:03d}", {"phase1_rounds": value}))
    for value in (10, 50, 100, 200, 500, 1000):
        specs.append(("trust", f"trust_{value:03d}", {"server_validation_size": value}))
    for value in (16, 32, 64, 128):
        specs.append(("latent", f"latent_{value:03d}", {"latent_dim": value}))
    return specs


def _robustness_specs() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        ("malicious_ratio", f"malicious_ratio_{value:.1f}".replace(".", "p"), {"num_malicious": int(100 * value)})
        for value in (0.2, 0.3, 0.4)
    ]


def _result_path(output_dir: Path, task: str, attack: str) -> Path:
    return output_dir / f"{task}__{attack}__svdd.json"


def _complete(path: Path, task: str, attack: str, rounds: int, seed: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        effective = meta.get("effective_config", {})
        records = payload.get("rounds")
        return (
            isinstance(records, list)
            and len(records) == rounds
            and meta.get("task") == task
            and meta.get("attack") == attack
            and meta.get("defense") == "svdd"
            and int(meta.get("total_rounds", -1)) == rounds
            and int(effective.get("seed", -1)) == seed
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _legacy_labels(overrides: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    malicious = int(overrides.get("num_malicious", 30))
    if malicious != 30:
        ratio_label = f"malicious_ratio_{malicious / 100:.1f}".replace(".", "p")
        return [ratio_label]
    # The legacy tree has no lambda-specific folders. Its baseline/factor
    # results are reusable only when the requested lambda is exactly 0.5.
    if abs(float(overrides.get("svdd_lambda", 0.5)) - 0.5) > 1e-12:
        return []
    labels.append("malicious_ratio_0p3")
    if "phase1_rounds" in overrides:
        labels.append(f"phase1_{int(overrides['phase1_rounds']):03d}")
    if "server_validation_size" in overrides:
        labels.append(f"trust_{int(overrides['server_validation_size']):03d}")
    if "latent_dim" in overrides:
        labels.append(f"latent_{int(overrides['latent_dim']):03d}")
    # The old baseline labels are also exact matches for the default config.
    labels.extend(("phase1_015", "trust_050", "latent_064"))
    return list(dict.fromkeys(labels))


def _write_config(
    root: Path,
    study: str,
    factor: str,
    label: str,
    task: str,
    attack: str,
    seed: int,
    rounds: int,
    factor_overrides: dict[str, Any],
    validation_tie_break: str = "largest",
) -> tuple[Path, Path, Path]:
    output_dir = root / study / factor / label / task / attack / f"seed_{seed}"
    config_path = root / "_configs" / study / factor / label / task / attack / f"seed_{seed}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update(factor_overrides)
    overrides.update({
        "seed": seed,
        "total_rounds": rounds,
        "svdd_validation_tie_break": validation_tie_break,
    })
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
    return config_path, output_dir, _result_path(output_dir, task, attack)


def _parse_csv(value: str, cast: type[str] | type[int]) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("log/svdd_report_matrix_absolute_mad_topk10_40_300"))
    parser.add_argument("--legacy-root", type=Path, default=Path("log/svdd_cross_task_sensitivity_absolute_mad_topk10_40_300"))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=10)
    parser.add_argument("--gpu-memory-budget-gb", type=float, default=20.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--resume-root", type=Path, default=None)
    parser.add_argument(
        "--skip-factors",
        default="",
        help="Comma-separated factor names to omit, e.g. latent",
    )
    parser.add_argument(
        "--skip-tasks",
        default="",
        help="Comma-separated task names to omit, e.g. ag_news",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--only-factors",
        default="",
        help="Comma-separated factor labels to run, e.g. trust_200",
    )
    parser.add_argument(
        "--svdd-validation-tie-break",
        choices=("largest", "smallest", "median"),
        default="largest",
        help="Tie rule for equal validation accuracy among Top-K candidates.",
    )
    args = parser.parse_args()

    skip_factors = {
        item.strip().lower()
        for item in args.skip_factors.split(",")
        if item.strip()
    }
    skip_tasks = {
        item.strip().lower()
        for item in args.skip_tasks.split(",")
        if item.strip()
    }
    only_factors = {
        item.strip().lower()
        for item in args.only_factors.split(",")
        if item.strip()
    }

    seeds = tuple(_parse_csv(args.seeds, int))
    gpus = tuple(_parse_csv(args.gpus, int))
    if (
        not seeds
        or not gpus
        or args.rounds < 1
        or args.workers_per_gpu < 1
        or args.gpu_memory_budget_gb <= 0.0
        or args.omp_threads < 1
    ):
        parser.error("seeds, gpus, rounds, workers-per-gpu, and omp-threads must be positive")

    root = args.output_root.resolve()
    legacy_root = args.legacy_root.resolve()
    resume_root = args.resume_root.resolve() if args.resume_root is not None else None
    if resume_root is not None and not resume_root.is_dir():
        parser.error(f"resume-root does not exist: {resume_root}")
    root.mkdir(parents=True, exist_ok=True)
    requested: list[dict[str, Any]] = []
    for study, specs in (("parameter_sensitivity", _parameter_specs()), ("robustness", _robustness_specs())):
        for factor, label, factor_overrides in specs:
            if only_factors and label.lower() not in only_factors:
                continue
            if factor.lower() in skip_factors or f"{study}:{factor}".lower() in skip_factors:
                continue
            for task in TASKS:
                if task.lower() in skip_tasks:
                    continue
                attacks = IMAGE_ATTACKS if task in IMAGE_TASKS else AG_NEWS_ATTACKS
                for attack in attacks:
                    for seed in seeds:
                        overrides = dict(BASE_OVERRIDES)
                        overrides.update(factor_overrides)
                        key = (task, attack, seed, tuple(sorted(overrides.items())))
                        config_path, output_dir, result_path = _write_config(
                            root,
                            study,
                            factor,
                            label,
                            task,
                            attack,
                            seed,
                            args.rounds,
                            factor_overrides,
                            args.svdd_validation_tie_break,
                        )
                        requested.append({
                            "study": study, "factor": factor, "label": label,
                            "task": task, "attack": attack, "seed": seed,
                            "key": key, "config_path": config_path,
                            "output_dir": output_dir, "result_path": result_path,
                            "status": "pending",
                        })

    canonical: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in requested:
        canonical.setdefault(item["key"], item)
    aliases: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in requested:
        aliases.setdefault(item["key"], []).append(item)

    jobs: list[dict[str, Any]] = []
    legacy_reused = 0
    resumed = 0
    complete = 0
    for key, item in canonical.items():
        result_path = item["result_path"]
        if not args.force and _complete(result_path, item["task"], item["attack"], args.rounds, item["seed"]):
            item["status"] = "complete"
            complete += 1
            continue
        if not args.force and resume_root is not None:
            relative = result_path.relative_to(root)
            source = resume_root / relative
            if source.exists() and _complete(source, item["task"], item["attack"], args.rounds, item["seed"]):
                result_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, result_path)
                item["status"] = "resumed"
                item["source_result"] = str(source)
                resumed += 1
                complete += 1
                continue
        if not args.force:
            overrides = dict(item["key"][3])
            for legacy_label in _legacy_labels(overrides):
                source = legacy_root / item["task"] / item["attack"] / legacy_label / f"seed_{item['seed']}" / f"{item['task']}__{item['attack']}__svdd.json"
                if source.exists() and _complete(source, item["task"], item["attack"], args.rounds, item["seed"]):
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    if result_path.exists() or result_path.is_symlink():
                        result_path.unlink()
                    result_path.symlink_to(source)
                    item["status"] = "legacy_reused"
                    item["source_result"] = str(source)
                    legacy_reused += 1
                    complete += 1
                    break
        if item["status"] == "pending":
            jobs.append(item)

    manifest = {
        "description": "Approved AE-SVDD sensitivity and malicious-ratio robustness matrix",
        "rounds": args.rounds,
        "seeds": seeds,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "gpu_memory_budget_gb": args.gpu_memory_budget_gb,
        "fixed": dict(BASE_OVERRIDES),
        "svdd_validation_tie_break": args.svdd_validation_tie_break,
        "topk_reject_ratios": [0.10, 0.20, 0.30, 0.40, 0.50],
        "requested_jobs": len(requested),
        "unique_jobs": len(canonical),
        "legacy_reused": legacy_reused,
        "resumed": resumed,
        "complete_before_run": complete,
        "pending_before_run": len(jobs),
        "jobs": [
            {k: str(v) if isinstance(v, Path) else v for k, v in item.items() if k not in {"key", "config_path", "output_dir", "result_path"}}
            for item in requested
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"requested={len(requested)} unique={len(canonical)} legacy_reused={legacy_reused} resumed={resumed} complete={complete} pending={len(jobs)}", flush=True)
    if args.prepare_only:
        return 0
    if not jobs:
        return 0

    # Use one queue per GPU. First interleave datasets globally, then assign
    # the interleaved stream round-robin. This makes the first batch on every
    # card contain both fast and slow datasets, rather than placing all early
    # MNIST/FashionMNIST jobs ahead of CIFAR10/AGNews.
    gpu_queues: dict[int, queue.Queue[dict[str, Any]]] = {
        gpu: queue.Queue() for gpu in gpus
    }
    jobs_by_task = {task: [] for task in TASKS}
    for job in jobs:
        jobs_by_task[job["task"]].append(job)
    interleaved: list[dict[str, Any]] = []
    for index in range(max(len(items) for items in jobs_by_task.values())):
        for task in TASKS:
            items = jobs_by_task[task]
            if index < len(items):
                interleaved.append(items[index])
    for index, job in enumerate(interleaved):
        gpu = gpus[index % len(gpus)]
        job["assigned_gpu"] = gpu
        gpu_queues[gpu].put(job)
    print(
        "gpu_pending=" + ",".join(
            f"{gpu}:{gpu_queues[gpu].qsize()}" for gpu in gpus
        ),
        flush=True,
    )
    failures: list[tuple[dict[str, Any], int]] = []
    lock = threading.Lock()
    gpu_conditions = {gpu: threading.Condition() for gpu in gpus}
    gpu_reserved_gb = {gpu: 0.0 for gpu in gpus}

    def reserve_gpu_memory(gpu: int, amount_gb: float) -> None:
        condition = gpu_conditions[gpu]
        with condition:
            while gpu_reserved_gb[gpu] + amount_gb > args.gpu_memory_budget_gb:
                condition.wait()
            gpu_reserved_gb[gpu] += amount_gb

    def release_gpu_memory(gpu: int, amount_gb: float) -> None:
        condition = gpu_conditions[gpu]
        with condition:
            gpu_reserved_gb[gpu] = max(0.0, gpu_reserved_gb[gpu] - amount_gb)
            condition.notify_all()

    def worker(gpu: int, worker_id: int) -> None:
        pending = gpu_queues[gpu]
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / "console.log"
            memory_estimate_gb = float(TASK_MEMORY_GB[job["task"]])
            reserve_gpu_memory(gpu, memory_estimate_gb)
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
            try:
                with lock:
                    print(f"START gpu={gpu} worker={worker_id} mem={memory_estimate_gb:.1f}GB {job['study']} {job['factor']}={job['label']} {job['task']}/{job['attack']}/seed_{job['seed']}", flush=True)
                with console_path.open("w", encoding="utf-8") as stream:
                    completed_process = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, check=False, start_new_session=True)
                with lock:
                    if completed_process.returncode == 0:
                        print(f"DONE {job['study']} {job['factor']}={job['label']} {job['task']}/{job['attack']}/seed_{job['seed']}", flush=True)
                    else:
                        failures.append((job, completed_process.returncode))
                        print(f"FAIL {job['study']} {job['factor']}={job['label']} {job['task']}/{job['attack']}/seed_{job['seed']} exit={completed_process.returncode} log={console_path}", flush=True)
            finally:
                release_gpu_memory(gpu, memory_estimate_gb)
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu, worker_id), daemon=False) for gpu in gpus for worker_id in range(args.workers_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"finished failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
