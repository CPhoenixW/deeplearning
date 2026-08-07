#!/usr/bin/env python3
"""Launch C002-based AE-SVDD alpha configurations with validation Top-K.

Each active variant is assigned to one GPU.  A variant worker runs all requested
seeds sequentially on its GPU, and each pipeline JSON contains the complete
attack matrix (clean, GN, LF, SF, BD, LIE, and Mix).  The runner is resumable:
completed seed directories are skipped unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


VARIANTS: dict[str, dict[str, Any]] = {
    "alpha02": {"alpha": 0.2},
    "alpha025": {"alpha": 0.25},
    "alpha033": {"alpha": 1.0 / 3.0},
    "alpha05": {"alpha": 0.5},
    "alpha075": {"alpha": 0.75},
    "alpha1": {"alpha": 1.0},
}

DEFAULT_ATTACKS = ("none", "gn", "lf", "sf", "bd", "lie", "mix")
DEFAULT_SEEDS = (42,)

# c002's effective configuration, including the stage-B population and
# runtime settings.  Variant-specific values are overlaid below.
C002_OVERRIDES: dict[str, Any] = {
    "num_clients": 100,
    "num_malicious": 30,
    "total_rounds": 300,
    "client_lr": 0.05,
    "client_momentum": 0.9,
    "client_weight_decay": 0.0001,
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
    "phase1_rounds": 15,
    "center_ema_decay": 0.9,
    "svdd_grad_clip": 1.0,
    "alpha": 0.5,
    "center_init_quantile": 0.5,
    "phase2_recon_quantile": 0.8,
    "svdd_feature_clip": 10.0,
    "device": "cuda",
}

# C002 was calibrated on CIFAR-10.  Keep its defense/runtime settings while
# restoring the task-model learning-rate calibration for other image tasks.
TASK_CLIENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "fashion_mnist": {"client_lr": 0.1, "client_weight_decay": 0.0},
    "mnist": {"client_lr": 0.1, "client_weight_decay": 0.0001},
    "ag_news": {"client_lr": 0.1, "client_weight_decay": 0.0},
}


def _parse_csv(value: str, *, cast: type = str) -> tuple[Any, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV argument must contain at least one value")
    return tuple(cast(item) for item in items)


def _json_config(
    *,
    task: str,
    variant: str,
    seed: int,
    output_dir: Path,
    attacks: Sequence[str],
    rounds: int,
) -> dict[str, Any]:
    overrides = dict(C002_OVERRIDES)
    overrides.update(TASK_CLIENT_OVERRIDES.get(task, {}))
    overrides.update(VARIANTS[variant])
    overrides.update({"seed": int(seed), "total_rounds": int(rounds)})
    return {
        "task": task,
        "attacks": ",".join(attacks),
        "defenses": "svdd",
        "log_dir": str(output_dir),
        "fed_config_file": "configs/federated.json",
        "hyperparameters_file": "configs/hyperparameters.json",
        "fed_config_overrides": overrides,
    }


def _write_configs(
    root: Path,
    *,
    task: str,
    variant: str,
    seeds: Sequence[int],
    attacks: Sequence[str],
    rounds: int,
) -> list[tuple[int, Path, Path]]:
    config_root = root / "_configs" / variant
    config_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, Path, Path]] = []
    for seed in seeds:
        output_dir = root / variant / f"seed_{seed}"
        config_path = config_root / f"seed_{seed}.json"
        payload = _json_config(
            task=task,
            variant=variant,
            seed=seed,
            output_dir=output_dir,
            attacks=attacks,
            rounds=rounds,
        )
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        jobs.append((int(seed), config_path, output_dir))
    return jobs


def _write_attack_configs(
    root: Path,
    *,
    task: str,
    variant: str,
    seed: int,
    output_dir: Path,
    attacks: Sequence[str],
    rounds: int,
) -> dict[str, Path]:
    config_root = root / "_configs" / variant / f"seed_{seed}"
    config_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for attack in attacks:
        path = config_root / f"{attack}.json"
        payload = _json_config(
            task=task,
            variant=variant,
            seed=seed,
            output_dir=output_dir,
            attacks=(attack,),
            rounds=rounds,
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths[str(attack)] = path
    return paths


def _result_path(output_dir: Path, task: str, attack: str) -> Path:
    return output_dir / f"{task}__{attack}__svdd.json"


def _attack_complete(
    path: Path, *, task: str, variant: str, rounds: int
) -> bool:
    """Return whether one final result has the expected run contract."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    recorded_rounds = payload.get("rounds", [])
    if not isinstance(recorded_rounds, list) or len(recorded_rounds) != int(rounds):
        return False
    meta = payload.get("meta", {})
    effective = meta.get("effective_config", {}) if isinstance(meta, dict) else {}
    if int(meta.get("total_rounds", -1)) != int(rounds):
        return False
    if str(meta.get("task", "")) != str(task):
        return False
    for key, value in VARIANTS[variant].items():
        if effective.get(key) != value:
            return False
    return True


def _complete(
    output_dir: Path, task: str, attacks: Sequence[str], rounds: int
) -> bool:
    variant = output_dir.parent.name
    return all(
        _attack_complete(
            _result_path(output_dir, task, attack),
            task=task,
            variant=variant,
            rounds=rounds,
        )
        for attack in attacks
    )


def _run_variant(
    *,
    root: Path,
    task: str,
    variant: str,
    gpu: int,
    seeds: Sequence[int],
    attacks: Sequence[str],
    rounds: int,
    python_bin: str,
    force: bool,
    attack_workers: int,
) -> int:
    variant_root = root / variant
    launcher_log = variant_root / "launcher.log"
    variant_root.mkdir(parents=True, exist_ok=True)
    with launcher_log.open("a", encoding="utf-8") as log:
        log.write(
            f"task={task} variant={variant} gpu={gpu} seeds={list(seeds)} "
            f"attacks={list(attacks)} rounds={rounds}\n"
        )
        for seed in seeds:
            output_dir = root / variant / f"seed_{seed}"
            if not force and _complete(output_dir, task, attacks, rounds):
                log.write(f"SKIP complete seed={seed}\n")
                log.flush()
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            config_paths = _write_attack_configs(
                root,
                task=task,
                variant=variant,
                seed=int(seed),
                output_dir=output_dir,
                attacks=attacks,
                rounds=rounds,
            )
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            pending = [
                attack
                for attack in attacks
                if force
                or not _attack_complete(
                    _result_path(output_dir, task, attack),
                    task=task,
                    variant=variant,
                    rounds=rounds,
                )
            ]
            log.write(
                f"SEED seed={seed} pending={pending} "
                f"attack_workers={attack_workers}\n"
            )
            log.flush()
            for start in range(0, len(pending), attack_workers):
                batch = pending[start : start + attack_workers]
                children: list[tuple[str, subprocess.Popen[Any], Any]] = []
                for attack in batch:
                    attack_log = (output_dir / f"{attack}.log").open(
                        "a", encoding="utf-8"
                    )
                    command = [
                        python_bin,
                        "-m",
                        "src.pipeline",
                        "--config",
                        str(config_paths[attack]),
                    ]
                    attack_log.write(
                        f"START seed={seed} attack={attack} "
                        f"gpu={gpu} command={' '.join(command)}\n"
                    )
                    attack_log.flush()
                    child = subprocess.Popen(
                        command,
                        cwd=str(Path(__file__).resolve().parents[1]),
                        env=env,
                        stdout=attack_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    children.append((attack, child, attack_log))
                for attack, child, attack_log in children:
                    code = child.wait()
                    attack_log.write(
                        f"END seed={seed} attack={attack} returncode={code}\n"
                    )
                    attack_log.close()
                    log.write(
                        f"END seed={seed} attack={attack} returncode={code}\n"
                    )
                    log.flush()
                    if code != 0:
                        return int(code)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("log/alpha_topk_sweep"),
        help="Output root for configs, logs, and result JSON files.",
    )
    parser.add_argument("--gpus", default="0,1,2", help="Three GPU ids in variant order.")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--task",
        default="cifar10",
        help="Task registry key (for example: cifar10 or fashion_mnist).",
    )
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument(
        "--attack-workers",
        type=int,
        default=6,
        help="Maximum simultaneous attack processes per GPU (default: 6).",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=".venv/bin/python",
        help="Python executable used for the pipeline.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun complete seeds.")
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "all"],
        default="all",
        help="Run one variant or all configured variants; all runs in GPU-sized waves.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after starting workers (only valid when one wave is sufficient).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = _parse_csv(args.gpus, cast=int)
    seeds = _parse_csv(args.seeds, cast=int)
    attacks = _parse_csv(args.attacks, cast=str)
    if args.variant == "all" and not gpus:
        parser.error("--gpus must provide at least one GPU id when --variant=all")
    if args.variant != "all" and not gpus:
        parser.error("--gpus must provide at least one GPU id")
    unknown = [attack for attack in attacks if attack not in DEFAULT_ATTACKS]
    if unknown:
        parser.error(f"Unsupported attack ids: {unknown}")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.attack_workers < 1:
        parser.error("--attack-workers must be positive")

    root = args.output_root.resolve()
    selected = list(VARIANTS) if args.variant == "all" else [args.variant]
    if args.variant == "all" and args.no_wait and len(selected) > len(gpus):
        parser.error(
            "--no-wait cannot supervise multiple waves; run the command itself "
            "under nohup without --no-wait"
        )
    for index, variant in enumerate(selected):
        gpu = int(gpus[index % len(gpus)])
        print(
            f"{variant}: task={args.task} gpu={gpu} seeds={list(seeds)} attacks={list(attacks)} "
            f"rounds={args.rounds} params={VARIANTS[variant]}"
        )
    if args.dry_run:
        return 0

    # A single variant runs directly. The all-variants mode executes
    # GPU-sized waves, so six variants on three GPUs become two waves.
    if args.variant != "all":
        return _run_variant(
            root=root,
            task=str(args.task),
            variant=args.variant,
            gpu=int(gpus[0]),
            seeds=seeds,
            attacks=attacks,
            rounds=args.rounds,
            python_bin=str(args.python_bin),
            force=bool(args.force),
            attack_workers=int(args.attack_workers),
        )

    root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    result = 0
    for wave_start in range(0, len(selected), len(gpus)):
        wave = selected[wave_start : wave_start + len(gpus)]
        workers: list[tuple[str, subprocess.Popen[Any], Any]] = []
        print(f"starting wave={1 + wave_start // len(gpus)} variants={wave}")
        for gpu_index, variant in enumerate(wave):
            gpu = int(gpus[gpu_index])
            variant_root = root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            supervisor_log = (variant_root / "supervisor.log").open(
                "a", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(script_path),
                "--variant",
                variant,
                "--task",
                str(args.task),
                "--gpus",
                str(gpu),
                "--seeds",
                ",".join(str(seed) for seed in seeds),
                "--attacks",
                ",".join(attacks),
                "--rounds",
                str(args.rounds),
                "--attack-workers",
                str(args.attack_workers),
                "--output-root",
                str(root),
                "--python",
                str(args.python_bin),
            ]
            if args.force:
                command.append("--force")
            process = subprocess.Popen(
                command,
                cwd=str(script_path.parents[1]),
                stdout=supervisor_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            workers.append((variant, process, supervisor_log))
            print(f"started {variant} gpu={gpu} pid={process.pid}")
        if args.no_wait:
            for _variant, _process, log in workers:
                log.close()
            return 0
        for variant, process, log in workers:
            code = process.wait()
            log.close()
            print(f"finished {variant} returncode={code}")
            if code and result == 0:
                result = int(code)
        if result:
            break
    return result


if __name__ == "__main__":
    raise SystemExit(main())
