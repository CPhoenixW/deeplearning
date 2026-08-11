#!/usr/bin/env python3
"""Run the RQ3 mechanism ablation matrix.

RQ3 compares FedAvg, P1-only, P2-only, and the full two-stage mechanism while
varying the malicious-client ratio.  The runner keeps the Phase 2 score and
loss coefficient fixed so mechanism and score/loss ablations are not mixed.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = ("fedavg", "p1_only", "p2_only", "full")
DEFAULT_ATTACKS = ("gn", "lie", "minmax", "minsum", "bd", "mix")
DEFAULT_RATIOS = (0.10, 0.20, 0.30, 0.40)
DEFAULT_SEEDS = (42,)

BASE_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "total_rounds": 100,
    "server_validation_size": 50,
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
    "client_batch_group_size": 2,
    "round_diagnostics": False,
    "dirichlet_alpha": 1.0,
    "hf_datasets_offline": True,
    "mixed_attack_types": "lf,bd,gn,sf,lie,minmax,minsum",
    "latent_dim": 64,
    "ae_lr": 0.001,
    "ae_weight_decay": 1e-6,
    "ae_grad_clip": 1.0,
    "svdd_input_mode": "delta",
    "svdd_feature_mode": "fixed_projection",
    "param_descriptor_dim": 4096,
    "param_descriptor_seed": 2027,
    "param_descriptor_global_ratio": 0.5,
    "param_descriptor_layer_ratio": 0.375,
    "param_descriptor_statistics_ratio": 0.125,
    "param_descriptor_device": "cuda",
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "svdd_feature_clip": 10.0,
    "phase1_score_mode": "recon",
    "phase2_score_mode": "svdd",
    "alpha": 0.5,
    "device": "cuda",
}


def _parse_csv(value: str, cast: type = str) -> tuple[Any, ...]:
    parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("CSV argument must not be empty")
    return parsed


def _parse_ratios(value: str) -> tuple[float, ...]:
    ratios = _parse_csv(value, float)
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("RQ3 malicious ratios must be in (0, 1)")
    return tuple(float(ratio) for ratio in ratios)


def _mechanism_overrides(mechanism: str, rounds: int) -> dict[str, Any]:
    if mechanism == "fedavg":
        return {"defense": "avg", "phase1_rounds": 15}
    if mechanism == "p1_only":
        return {"defense": "svdd", "phase1_rounds": int(rounds)}
    if mechanism == "p2_only":
        return {"defense": "svdd", "phase1_rounds": 0}
    if mechanism == "full":
        return {"defense": "svdd", "phase1_rounds": 15}
    raise ValueError(f"Unknown RQ3 mechanism: {mechanism}")


def iter_specs(
    *,
    mechanisms: Iterable[str],
    attacks: Iterable[str],
    ratios: Iterable[float],
    seeds: Iterable[int],
) -> Iterable[tuple[str, str, float, int]]:
    """Yield clean control at 0% plus attack conditions at each ratio."""

    for seed in seeds:
        for mechanism in mechanisms:
            yield mechanism, "none", 0.0, int(seed)
            for ratio in ratios:
                for attack in attacks:
                    yield mechanism, attack, float(ratio), int(seed)


def _result_path(output_dir: Path, task: str, attack: str, mechanism: str) -> Path:
    defense = _mechanism_overrides(mechanism, 1)["defense"]
    return output_dir / f"{task}__{attack}__{defense}.json"


def _complete(
    path: Path,
    *,
    task: str,
    attack: str,
    mechanism: str,
    ratio: float,
    seed: int,
    rounds: int,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    records = payload.get("rounds")
    meta = payload.get("meta")
    if not isinstance(records, list) or len(records) != int(rounds) or not isinstance(meta, dict):
        return False
    effective = meta.get("effective_config", {})
    if not isinstance(effective, dict):
        return False
    expected_defense = _mechanism_overrides(mechanism, rounds)["defense"]
    expected_malicious = int(round(100 * ratio))
    if attack == "none":
        expected_malicious = 0
    return (
        str(meta.get("task", "")) == task
        and str(meta.get("attack", "")) == attack
        and str(meta.get("defense", "")) == expected_defense
        and int(meta.get("total_rounds", -1)) == int(rounds)
        and int(effective.get("seed", -1)) == int(seed)
        and int(effective.get("phase1_rounds", -1))
        == int(_mechanism_overrides(mechanism, rounds)["phase1_rounds"])
        and str(effective.get("phase1_score_mode", "")) == "recon"
        and str(effective.get("phase2_score_mode", "")) == "svdd"
        and abs(float(effective.get("alpha", -1.0)) - 0.5) < 1e-8
        and int(effective.get("num_clients", -1)) - int(effective.get("num_benign", -1))
        == expected_malicious
    )


def _write_config(
    root: Path,
    *,
    task: str,
    mechanism: str,
    attack: str,
    ratio: float,
    seed: int,
    rounds: int,
) -> tuple[Path, Path]:
    ratio_id = int(round(100 * ratio))
    output_dir = root / mechanism / attack / f"mal_{ratio_id:02d}" / f"seed_{seed}"
    config_path = root / "_configs" / mechanism / attack / f"mal_{ratio_id:02d}" / f"seed_{seed}.json"
    overrides = dict(BASE_OVERRIDES)
    overrides.update(
        {
            "seed": int(seed),
            "total_rounds": int(rounds),
            "num_malicious": int(ratio_id),
            "mixed_attack_types": "lf,bd,gn,sf,lie,minmax,minsum",
        }
    )
    overrides.update({key: value for key, value in _mechanism_overrides(mechanism, rounds).items() if key != "defense"})
    if attack == "none":
        overrides["num_malicious"] = 0
    payload = {
        "task": task,
        "attacks": attack,
        "defenses": _mechanism_overrides(mechanism, rounds)["defense"],
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path, output_dir


def build_jobs(
    root: Path,
    *,
    task: str,
    mechanisms: tuple[str, ...],
    attacks: tuple[str, ...],
    ratios: tuple[float, ...],
    seeds: tuple[int, ...],
    rounds: int,
    force: bool = False,
    max_jobs: int | None = None,
) -> list[tuple[str, str, float, int, Path, Path]]:
    jobs: list[tuple[str, str, float, int, Path, Path]] = []
    for mechanism, attack, ratio, seed in iter_specs(
        mechanisms=mechanisms, attacks=attacks, ratios=ratios, seeds=seeds
    ):
        config_path, output_dir = _write_config(
            root,
            task=task,
            mechanism=mechanism,
            attack=attack,
            ratio=ratio,
            seed=seed,
            rounds=rounds,
        )
        result_path = _result_path(output_dir, task, attack, mechanism)
        if force or not _complete(
            result_path,
            task=task,
            attack=attack,
            mechanism=mechanism,
            ratio=ratio,
            seed=seed,
            rounds=rounds,
        ):
            jobs.append((mechanism, attack, ratio, seed, config_path, output_dir))
            if max_jobs is not None and len(jobs) >= int(max_jobs):
                break
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="fashion_mnist")
    parser.add_argument("--mechanisms", default=",".join(MECHANISMS))
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--malicious-ratios", default=",".join(map(str, DEFAULT_RATIOS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("log/rq3_mechanism_fashion_mnist_20260811"))
    parser.add_argument("--python", dest="python_bin", default=".venv/bin/python")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mechanisms = tuple(item.strip().lower() for item in _parse_csv(args.mechanisms))
    attacks = tuple(item.strip().lower() for item in _parse_csv(args.attacks))
    ratios = _parse_ratios(args.malicious_ratios)
    seeds = tuple(int(item) for item in _parse_csv(args.seeds, int))
    gpus = tuple(int(item) for item in _parse_csv(args.gpus, int))
    if args.rounds < 20:
        parser.error("RQ3 mechanism runs must use at least 20 rounds")
    if any(item not in MECHANISMS for item in mechanisms):
        parser.error(f"Unsupported mechanisms: {sorted(set(mechanisms) - set(MECHANISMS))}")
    if "bd" in attacks or "mix" in attacks:
        if args.task == "ag_news":
            parser.error("AG News does not support image-trigger BD/Mix in RQ3")
    if args.workers_per_gpu < 1 or not gpus:
        parser.error("at least one GPU and one worker per GPU are required")
    root = args.output_root.resolve()
    jobs = build_jobs(
        root,
        task=str(args.task),
        mechanisms=mechanisms,
        attacks=attacks,
        ratios=ratios,
        seeds=seeds,
        rounds=int(args.rounds),
        force=bool(args.force),
        max_jobs=args.max_jobs,
    )
    expected = len(mechanisms) * len(seeds) * (1 + len(attacks) * len(ratios))
    print(f"jobs={len(jobs)} expected={expected} task={args.task} rounds={args.rounds} gpus={list(gpus)} workers_per_gpu={args.workers_per_gpu}")
    for mechanism, attack, ratio, seed, config, _output in jobs[:80]:
        print(f"PENDING mechanism={mechanism} attack={attack} ratio={ratio:.2f} seed={seed} config={config}")
    if len(jobs) > 80:
        print(f"... {len(jobs) - 80} more jobs")
    if args.dry_run or not jobs:
        return 0
    pending: queue.Queue[tuple[str, str, float, int, Path, Path]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, float, int, int]] = []
    lock = threading.Lock()

    def worker(gpu: int, worker_id: int) -> None:
        while True:
            try:
                mechanism, attack, ratio, seed, config, output_dir = pending.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console = output_dir / "run.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": str(int(args.omp_threads)),
                    "MKL_NUM_THREADS": str(int(args.omp_threads)),
                    "OPENBLAS_NUM_THREADS": str(int(args.omp_threads)),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [str(args.python_bin), "-u", "-m", "src.pipeline", "--config", str(config)]
            with lock:
                print(f"START gpu={gpu} worker={worker_id} mechanism={mechanism} attack={attack} ratio={ratio:.2f} seed={seed}", flush=True)
            with console.open("a", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, check=False, start_new_session=True)
            with lock:
                if completed.returncode == 0:
                    print(f"DONE mechanism={mechanism} attack={attack} ratio={ratio:.2f} seed={seed}", flush=True)
                else:
                    failures.append((mechanism, attack, ratio, seed, int(completed.returncode)))
                    print(f"FAIL mechanism={mechanism} attack={attack} ratio={ratio:.2f} seed={seed} exit={completed.returncode}", flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu, worker_id), daemon=False) for gpu in gpus for worker_id in range(args.workers_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
