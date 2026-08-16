#!/usr/bin/env python3
"""Generate, schedule, resume and rank Stage-A clean FedAvg calibrations.

This utility contains no training implementation. Every trial is materialized as
an ordinary pipeline JSON and executed through ``python -m src.pipeline``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TASKS = {"mnist", "fashion_mnist", "cifar10", "covid19", "ag_news"}


def _load_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_object(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    path.write_text(rendered, encoding="utf-8")


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _number_slug(value: float) -> str:
    text = format(float(value), ".10g")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _parse_csv_ints(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer")
    return values


def load_manifest(path: str | Path) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = _resolve(path)
    manifest = _load_object(manifest_path)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Stage-A manifest requires a non-empty tasks list")
    unknown_tasks = sorted(set(str(task) for task in tasks) - SUPPORTED_TASKS)
    if unknown_tasks:
        raise ValueError(f"Unsupported Stage-A tasks: {unknown_tasks}")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Stage-A manifest requires at least one seed")

    if "task_candidates" not in manifest:
        learning_rates = manifest.get("client_lrs")
        weight_decays = manifest.get("client_weight_decays")
        if not isinstance(learning_rates, list) or not learning_rates:
            raise ValueError("Stage-A manifest requires client_lrs")
        if not isinstance(weight_decays, list) or not weight_decays:
            raise ValueError("Stage-A manifest requires client_weight_decays")
        if any(float(value) <= 0.0 for value in learning_rates):
            raise ValueError("Every client_lr must be positive")
        if any(float(value) < 0.0 for value in weight_decays):
            raise ValueError("Every client_weight_decay must be non-negative")

    common = manifest.get("common_overrides", {})
    if not isinstance(common, dict):
        raise ValueError("common_overrides must be a JSON object")
    if int(common.get("total_rounds", 0)) < 1:
        raise ValueError("common_overrides.total_rounds must be at least 1")
    return manifest_path, manifest


def _candidate_pairs(manifest: Dict[str, Any], task: str) -> List[Tuple[float, float]]:
    task_candidates = manifest.get("task_candidates")
    if task_candidates is None:
        return [
            (float(learning_rate), float(weight_decay))
            for learning_rate in manifest["client_lrs"]
            for weight_decay in manifest["client_weight_decays"]
        ]
    if not isinstance(task_candidates, dict):
        raise ValueError("task_candidates must be a JSON object")
    candidates = task_candidates.get(task)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No promoted candidates configured for task {task!r}")
    pairs: List[Tuple[float, float]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"Invalid candidate for task {task!r}: {candidate!r}")
        pairs.append(
            (
                float(candidate["client_lr"]),
                float(candidate["client_weight_decay"]),
            )
        )
    return pairs


@dataclass(frozen=True)
class Trial:
    task: str
    client_lr: float
    client_weight_decay: float
    seed: int
    config_path: Path
    output_dir: Path

    @property
    def result_path(self) -> Path:
        return self.output_dir / f"{self.task}__none__avg.json"


def build_trials(manifest: Dict[str, Any]) -> List[Trial]:
    generated_root = _resolve(
        str(manifest.get("generated_config_dir", "configs/stage_a/generated"))
    )
    output_root = _resolve(str(manifest.get("output_root", "log/stage_a")))
    trials: List[Trial] = []
    for task_value in manifest["tasks"]:
        task = str(task_value)
        for learning_rate, weight_decay in _candidate_pairs(manifest, task):
            pair_name = (
                f"lr_{_number_slug(learning_rate)}__"
                f"wd_{_number_slug(weight_decay)}"
            )
            for seed_value in manifest["seeds"]:
                seed = int(seed_value)
                trial_name = f"{task}__{pair_name}__seed_{seed}"
                trials.append(
                    Trial(
                        task=task,
                        client_lr=learning_rate,
                        client_weight_decay=weight_decay,
                        seed=seed,
                        config_path=generated_root / task / f"{trial_name}.json",
                        output_dir=output_root / task / pair_name / f"seed_{seed}",
                    )
                )
    return trials


def _pipeline_payload(manifest: Dict[str, Any], trial: Trial) -> Dict[str, Any]:
    overrides = dict(manifest.get("common_overrides", {}))
    task_overrides = manifest.get("task_overrides", {})
    if isinstance(task_overrides, dict):
        selected = task_overrides.get(trial.task, {})
        if not isinstance(selected, dict):
            raise ValueError(f"task_overrides.{trial.task} must be an object")
        overrides.update(selected)
    overrides.update(
        {
            "attack_type": "none",
            "client_lr": trial.client_lr,
            "client_weight_decay": trial.client_weight_decay,
            "defense_type": "avg",
            "device": "cuda",
            "num_malicious": 0,
            "seed": trial.seed,
        }
    )
    return {
        "task": trial.task,
        "attacks": "none",
        "defenses": "avg",
        "log_dir": str(trial.output_dir),
        "fed_config_file": str(
            manifest.get("fed_config_file", "configs/federated.json")
        ),
        "hyperparameters_file": str(
            manifest.get("hyperparameters_file", "configs/hyperparameters.json")
        ),
        "fed_config_overrides": overrides,
    }


def generate_configs(manifest: Dict[str, Any]) -> List[Trial]:
    trials = build_trials(manifest)
    for trial in trials:
        _write_object(trial.config_path, _pipeline_payload(manifest, trial))
    return trials


def prepare_datasets(manifest: Dict[str, Any]) -> None:
    """Download/cache each task serially before concurrent workers start."""

    from src.config import (
        FedConfig,
        apply_fed_config_overrides,
        load_fed_config_values,
    )
    from src.tasks import get_task

    fed_config_path = _resolve(
        str(manifest.get("fed_config_file", "configs/federated.json"))
    )
    base_values = load_fed_config_values(fed_config_path)
    task_overrides = manifest.get("task_overrides", {})
    for task_value in manifest["tasks"]:
        task_name = str(task_value)
        config = FedConfig()
        apply_fed_config_overrides(config, base_values, source=str(fed_config_path))
        apply_fed_config_overrides(
            config,
            manifest.get("common_overrides", {}),
            source="stage_a.common_overrides",
        )
        apply_fed_config_overrides(
            config,
            manifest.get("prepare_data_overrides", {}),
            source="stage_a.prepare_data_overrides",
        )
        if isinstance(task_overrides, dict):
            apply_fed_config_overrides(
                config,
                task_overrides.get(task_name, {}),
                source=f"stage_a.task_overrides.{task_name}",
            )
        config.task_name = task_name
        task = get_task(config)
        config.num_classes = task.num_classes
        print(f"PREPARE {task_name} data_root={config.data_root}")
        client_loaders, _validation_loader, test_loader = task.build_dataloaders(config)
        train_samples = sum(len(loader.dataset) for loader in client_loaders)
        print(
            f"READY   {task_name} train={train_samples} "
            f"test={len(test_loader.dataset)} clients={len(client_loaders)}"
        )


def _is_complete(trial: Trial, expected_rounds: int) -> bool:
    if not trial.result_path.exists() or not trial.config_path.exists():
        return False
    try:
        payload = _load_object(trial.result_path)
        trial_config = _load_object(trial.config_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    meta = payload.get("meta", {})
    rounds = payload.get("rounds", [])
    if not isinstance(meta, dict) or not isinstance(rounds, list):
        return False
    effective = meta.get("effective_config", {})
    if not isinstance(effective, dict):
        return False
    expected_effective = trial_config.get("fed_config_overrides", {})
    if not isinstance(expected_effective, dict):
        return False
    return (
        meta.get("task") == trial.task
        and meta.get("attack") == "none"
        and meta.get("defense") == "avg"
        and len(rounds) == expected_rounds
        and int(meta.get("total_rounds", -1)) == expected_rounds
        and all(
            effective.get(key, meta.get(key)) == value
            for key, value in expected_effective.items()
        )
    )


def _python_binary(manifest: Dict[str, Any]) -> str:
    configured = manifest.get("python")
    if configured:
        return str(configured)
    bundled = PROJECT_ROOT / "dl" / "bin" / "python"
    return str(bundled) if bundled.is_file() else sys.executable


def run_trials(
    manifest: Dict[str, Any],
    trials: Sequence[Trial],
    *,
    worker_gpu_ids: Sequence[int],
    force: bool = False,
) -> None:
    if not worker_gpu_ids:
        raise ValueError("At least one GPU worker is required")
    expected_rounds = int(manifest["common_overrides"]["total_rounds"])
    pending = [
        trial
        for trial in trials
        if force or not _is_complete(trial, expected_rounds)
    ]
    skipped = len(trials) - len(pending)
    print(
        f"Stage-A trials total={len(trials)} pending={len(pending)} "
        f"complete={skipped} workers={len(worker_gpu_ids)} "
        f"gpus={sorted(set(worker_gpu_ids))}"
    )
    if not pending:
        return

    work_queue: queue.Queue[Trial] = queue.Queue()
    for trial in pending:
        work_queue.put(trial)
    failures: List[Tuple[Trial, int]] = []
    lock = threading.Lock()
    python_binary = _python_binary(manifest)
    cpu_threads = str(int(manifest.get("cpu_threads_per_worker", 8)))

    def worker(worker_id: int, gpu_id: int) -> None:
        while True:
            try:
                trial = work_queue.get_nowait()
            except queue.Empty:
                return
            trial.output_dir.mkdir(parents=True, exist_ok=True)
            console_path = trial.output_dir / "console.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "MKL_NUM_THREADS": cpu_threads,
                    "OMP_NUM_THREADS": cpu_threads,
                    "OPENBLAS_NUM_THREADS": cpu_threads,
                    "PYTHONUNBUFFERED": "1",
                }
            )
            with lock:
                print(
                    f"[GPU {gpu_id}/W{worker_id}] START {trial.task} "
                    f"lr={trial.client_lr:g} wd={trial.client_weight_decay:g} "
                    f"seed={trial.seed}"
                )
            with console_path.open("a", encoding="utf-8") as console:
                completed = subprocess.run(
                    [
                        python_binary,
                        "-m",
                        "src.pipeline",
                        "--config",
                        str(trial.config_path),
                    ],
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                if completed.returncode == 0:
                    print(
                        f"[GPU {gpu_id}/W{worker_id}] DONE  "
                        f"{trial.config_path.name}"
                    )
                else:
                    failures.append((trial, completed.returncode))
                    print(
                        f"[GPU {gpu_id}/W{worker_id}] FAIL  "
                        f"{trial.config_path.name} "
                        f"exit={completed.returncode} log={console_path}"
                    )
            work_queue.task_done()

    workers = [
        threading.Thread(target=worker, args=(worker_id, gpu_id), daemon=False)
        for worker_id, gpu_id in enumerate(worker_gpu_ids)
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures:
        summary = ", ".join(
            f"{trial.config_path.name}:{return_code}"
            for trial, return_code in failures
        )
        raise RuntimeError(f"Stage-A trials failed: {summary}")


def _trial_score(trial: Trial, last_n: int) -> float:
    payload = _load_object(trial.result_path)
    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"No rounds in {trial.result_path}")
    values: List[float] = []
    for item in rounds[-last_n:]:
        evaluation = item.get("evaluation", {}) if isinstance(item, dict) else {}
        accuracy = evaluation.get("accuracy") if isinstance(evaluation, dict) else None
        if isinstance(accuracy, (int, float)):
            values.append(float(accuracy))
    if not values:
        raise ValueError(f"No clean TACC values in {trial.result_path}")
    return statistics.mean(values)


def select_candidates(
    manifest: Dict[str, Any],
    trials: Sequence[Trial],
    *,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    last_n = int(manifest.get("score_last_n_rounds", 10))
    expected_rounds = int(manifest["common_overrides"]["total_rounds"])
    expected_seeds = {int(seed) for seed in manifest["seeds"]}
    grouped: Dict[Tuple[str, float, float], Dict[int, float]] = {}
    missing: List[str] = []
    for trial in trials:
        complete_result = _is_complete(trial, expected_rounds)
        if not complete_result and not allow_incomplete:
            missing.append(
                f"{trial.result_path}: missing, incomplete, or stale effective config"
            )
            continue
        if not trial.result_path.exists():
            missing.append(str(trial.result_path))
            continue
        try:
            score = _trial_score(trial, last_n)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            missing.append(f"{trial.result_path}: {exc}")
            continue
        key = (trial.task, trial.client_lr, trial.client_weight_decay)
        grouped.setdefault(key, {})[trial.seed] = score

    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Stage-A selection requires all trials; missing/invalid={len(missing)}. "
            "Use --allow-incomplete only for provisional rankings."
        )

    rankings: Dict[str, List[Dict[str, Any]]] = {
        str(task): [] for task in manifest["tasks"]
    }
    for (task, learning_rate, weight_decay), scores_by_seed in grouped.items():
        complete = set(scores_by_seed) == expected_seeds
        if not complete and not allow_incomplete:
            continue
        seed_scores = [scores_by_seed[seed] for seed in sorted(scores_by_seed)]
        rankings[task].append(
            {
                "client_lr": learning_rate,
                "client_weight_decay": weight_decay,
                "mean_tacc": statistics.mean(seed_scores),
                "std_tacc": (
                    statistics.pstdev(seed_scores) if len(seed_scores) > 1 else 0.0
                ),
                "seed_tacc": {
                    str(seed): scores_by_seed[seed] for seed in sorted(scores_by_seed)
                },
                "complete": complete,
            }
        )

    selected: Dict[str, Dict[str, float]] = {}
    for task, rows in rankings.items():
        rows.sort(
            key=lambda row: (
                -float(row["mean_tacc"]),
                float(row["std_tacc"]),
                float(row["client_lr"]),
                float(row["client_weight_decay"]),
            )
        )
        if rows:
            selected[task] = {
                "client_lr": float(rows[0]["client_lr"]),
                "client_weight_decay": float(rows[0]["client_weight_decay"]),
            }

    return {
        "protocol": "clean FedAvg Stage-A task-model calibration",
        "score": f"mean clean TACC over each seed's last {last_n} rounds",
        "selection_uses_only": "clean_tacc",
        "expected_seeds": sorted(expected_seeds),
        "missing_or_invalid": missing,
        "selected_by_task": selected,
        "recommended_hyperparameters_patch": {"tasks": selected},
        "rankings": rankings,
    }


def promote_manifest(
    source: Dict[str, Any],
    selection: Dict[str, Any],
    *,
    top_k: int,
    rounds: int,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    promoted = copy.deepcopy(source)
    promoted["name"] = f"{source.get('name', 'stage_a')}_confirm"
    promoted.pop("client_lrs", None)
    promoted.pop("client_weight_decays", None)
    rankings = selection.get("rankings", {})
    task_candidates: Dict[str, List[Dict[str, float]]] = {}
    for task_value in source["tasks"]:
        task = str(task_value)
        rows = rankings.get(task, []) if isinstance(rankings, dict) else []
        if not isinstance(rows, list) or len(rows) < top_k:
            raise ValueError(f"Task {task!r} has fewer than {top_k} ranked candidates")
        task_candidates[task] = [
            {
                "client_lr": float(row["client_lr"]),
                "client_weight_decay": float(row["client_weight_decay"]),
            }
            for row in rows[:top_k]
        ]
    promoted["task_candidates"] = task_candidates
    promoted["seeds"] = [int(seed) for seed in seeds]
    promoted.setdefault("common_overrides", {})["total_rounds"] = int(rounds)
    defaults = source.get("promotion_defaults", {})
    promoted["generated_config_dir"] = str(
        defaults.get(
            "generated_config_dir", "configs/stage_a/generated_confirm"
        )
    )
    promoted["output_root"] = str(
        defaults.get("output_root", "log/stage_a/confirm")
    )
    promoted["selection_file"] = str(
        defaults.get("selection_file", "log/stage_a/confirm_selection.json")
    )
    return promoted


def _manifest_gpu_ids(manifest: Dict[str, Any], override: str | None) -> List[int]:
    if override:
        return _parse_csv_ints(override)
    values = manifest.get("gpus", [0, 1, 2])
    if not isinstance(values, list) or not values:
        raise ValueError("Manifest gpus must be a non-empty list")
    return [int(value) for value in values]


def _worker_gpu_ids(
    manifest: Dict[str, Any],
    gpu_override: str | None,
    workers_per_gpu_override: int | None,
    *,
    task: str | None = None,
) -> List[int]:
    gpu_ids = _manifest_gpu_ids(manifest, gpu_override)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU ids must be unique; use workers_per_gpu for concurrency")
    configured_workers: Any = manifest.get("workers_per_gpu", 1)
    per_task = manifest.get("workers_per_gpu_by_task", {})
    if task is not None and isinstance(per_task, dict):
        configured_workers = per_task.get(task, configured_workers)
    workers_per_gpu = int(
        workers_per_gpu_override
        if workers_per_gpu_override is not None
        else configured_workers
    )
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(workers_per_gpu)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-GPU Stage-A clean FedAvg calibration utility"
    )
    parser.add_argument("--manifest", required=True, help="Stage-A manifest JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Materialize one pipeline JSON per trial")
    subparsers.add_parser("plan", help="Print the validated trial plan")
    subparsers.add_parser(
        "prepare-data",
        help="Download/cache or validate all configured datasets before multi-GPU execution",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run/resume trials with configurable workers per GPU"
    )
    run_parser.add_argument("--gpus", help="Comma-separated GPU ids; defaults to manifest")
    run_parser.add_argument(
        "--workers-per-gpu",
        type=int,
        help="Concurrent trial processes per GPU; defaults to manifest",
    )
    run_parser.add_argument("--limit", type=int, help="Run only the first N trials")
    run_parser.add_argument("--force", action="store_true", help="Rerun complete trials")

    select_parser = subparsers.add_parser("select", help="Rank candidates using clean TACC")
    select_parser.add_argument("--allow-incomplete", action="store_true")

    promote_parser = subparsers.add_parser(
        "promote", help="Create a full-budget confirmation manifest from top candidates"
    )
    promote_parser.add_argument("--selection", help="Screening selection JSON")
    promote_parser.add_argument("--output", required=True, help="Confirmation manifest path")
    promote_parser.add_argument("--top-k", type=int)
    promote_parser.add_argument("--rounds", type=int)
    promote_parser.add_argument("--seeds", help="Comma-separated confirmation seeds")

    args = parser.parse_args()
    _manifest_path, manifest = load_manifest(args.manifest)
    trials = generate_configs(manifest)

    if args.command == "generate":
        print(f"Generated {len(trials)} pipeline JSON files")
        return
    if args.command == "plan":
        task_workers = {
            str(task): len(
                _worker_gpu_ids(manifest, None, None, task=str(task))
            )
            for task in manifest["tasks"]
        }
        print(
            f"Stage-A plan trials={len(trials)} tasks={len(manifest['tasks'])} "
            f"gpus={_manifest_gpu_ids(manifest, None)} "
            f"task_workers={task_workers}"
        )
        for task in manifest["tasks"]:
            count = sum(1 for trial in trials if trial.task == task)
            print(f"{task}: {count} trials")
        return
    if args.command == "prepare-data":
        prepare_datasets(manifest)
        return
    if args.command == "run":
        selected_trials = trials[: args.limit] if args.limit else trials
        per_task = manifest.get("workers_per_gpu_by_task")
        if args.workers_per_gpu is not None or not isinstance(per_task, dict):
            run_trials(
                manifest,
                selected_trials,
                worker_gpu_ids=_worker_gpu_ids(
                    manifest, args.gpus, args.workers_per_gpu
                ),
                force=bool(args.force),
            )
        else:
            for task in manifest["tasks"]:
                task_trials = [
                    trial for trial in selected_trials if trial.task == str(task)
                ]
                if not task_trials:
                    continue
                print(f"Stage-A task={task}")
                run_trials(
                    manifest,
                    task_trials,
                    worker_gpu_ids=_worker_gpu_ids(
                        manifest, args.gpus, None, task=str(task)
                    ),
                    force=bool(args.force),
                )
        return
    if args.command == "select":
        selection = select_candidates(
            manifest,
            trials,
            allow_incomplete=bool(args.allow_incomplete),
        )
        output_path = _resolve(
            str(manifest.get("selection_file", "log/stage_a/selection.json"))
        )
        _write_object(output_path, selection)
        print(f"Saved Stage-A ranking: {output_path}")
        for task, selected in selection["selected_by_task"].items():
            print(
                f"{task}: client_lr={selected['client_lr']:g} "
                f"client_weight_decay={selected['client_weight_decay']:g}"
            )
        return
    if args.command == "promote":
        defaults = manifest.get("promotion_defaults", {})
        selection_path = _resolve(
            args.selection
            or str(manifest.get("selection_file", "log/stage_a/selection.json"))
        )
        selection = _load_object(selection_path)
        top_k = int(args.top_k or defaults.get("top_k", 3))
        rounds = int(args.rounds or defaults.get("total_rounds", 300))
        seeds = (
            _parse_csv_ints(args.seeds)
            if args.seeds
            else [int(seed) for seed in defaults.get("seeds", [42, 43, 44])]
        )
        promoted = promote_manifest(
            manifest,
            selection,
            top_k=top_k,
            rounds=rounds,
            seeds=seeds,
        )
        output_path = _resolve(args.output)
        _write_object(output_path, promoted)
        print(f"Saved confirmation manifest: {output_path}")
        return


if __name__ == "__main__":
    main()
