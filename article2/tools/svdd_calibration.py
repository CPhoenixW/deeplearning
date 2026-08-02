from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_PARAMETER_KEYS = {
    "param_descriptor_dim",
    "latent_dim",
    "phase1_rounds",
    "ae_warmup_keep_ratio",
    "ae_lr",
}
DEFENSE_PARAMETER_KEYS = {
    "tau_start",
    "tau_end",
    "tau_anneal_rounds",
    "center_ema_decay",
    "svdd_loss_weight",
    "recon_loss_weight",
    "svdd_feature_clip",
}
SEARCH_PARAMETER_KEYS = COMMON_PARAMETER_KEYS | DEFENSE_PARAMETER_KEYS


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _read_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _values(manifest: Mapping[str, Any], key: str) -> List[Any]:
    space = manifest.get("parameter_space", {})
    values = space.get(key) if isinstance(space, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError(f"parameter_space.{key} must be a non-empty array")
    return values


def _validate_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = set(parameters) - SEARCH_PARAMETER_KEYS
    if unknown:
        raise ValueError(f"Unknown AE-SVDD search parameters: {sorted(unknown)}")
    missing = SEARCH_PARAMETER_KEYS - set(parameters)
    if missing:
        raise ValueError(f"Missing AE-SVDD search parameters: {sorted(missing)}")
    result = dict(parameters)
    if int(result["param_descriptor_dim"]) < 64:
        raise ValueError("param_descriptor_dim must be at least 64")
    if not 1 <= int(result["latent_dim"]) <= 64:
        raise ValueError("latent_dim must be in [1, 64]")
    if int(result["phase1_rounds"]) < 0:
        raise ValueError("phase1_rounds must be non-negative")
    if not 0.0 < float(result["ae_warmup_keep_ratio"]) <= 1.0:
        raise ValueError("ae_warmup_keep_ratio must be in (0, 1]")
    if float(result["ae_lr"]) <= 0.0:
        raise ValueError("ae_lr must be positive")
    if float(result["tau_end"]) > float(result["tau_start"]):
        raise ValueError("tau_end must be less than or equal to tau_start")
    if float(result["tau_end"]) <= 0.0:
        raise ValueError("tau_start and tau_end must be positive")
    if int(result["tau_anneal_rounds"]) < 1:
        raise ValueError("tau_anneal_rounds must be at least 1")
    if not 0.0 <= float(result["center_ema_decay"]) < 1.0:
        raise ValueError("center_ema_decay must be in [0, 1)")
    if float(result["svdd_loss_weight"]) <= 0.0 or float(
        result["recon_loss_weight"]
    ) <= 0.0:
        raise ValueError("SVDD and reconstruction loss weights must be positive")
    if float(result["svdd_feature_clip"]) <= 0.0:
        raise ValueError("svdd_feature_clip must be positive")
    return result


def _parameter_dimensions(manifest: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    scalar_keys = [
        "param_descriptor_dim",
        "latent_dim",
        "phase1_rounds",
        "ae_warmup_keep_ratio",
        "ae_lr",
        "tau_anneal_rounds",
        "center_ema_decay",
        "svdd_feature_clip",
    ]
    dimensions = {
        key: [{key: value} for value in _values(manifest, key)]
        for key in scalar_keys
    }
    tau_pairs = [
        {"tau_start": start, "tau_end": end}
        for start in _values(manifest, "tau_start")
        for end in _values(manifest, "tau_end")
        if float(end) <= float(start)
    ]
    if not tau_pairs:
        raise ValueError("No valid tau_start/tau_end pairs")
    dimensions["tau_schedule"] = tau_pairs
    dimensions["loss_ratio"] = [
        {
            "svdd_loss_weight": float(ratio),
            "recon_loss_weight": 1.0,
        }
        for ratio in _values(manifest, "loss_weight_ratio")
    ]
    return dimensions


def _signature(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def build_candidates(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    explicit = manifest.get("explicit_candidates")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("explicit_candidates must be a non-empty array")
        return [_validate_parameters(item) for item in explicit]

    count = int(manifest.get("num_candidates", 16))
    if count < 1:
        raise ValueError("num_candidates must be at least 1")
    baseline = _validate_parameters(manifest.get("baseline_parameters", {}))
    dimensions = _parameter_dimensions(manifest)
    seed = int(manifest.get("design_seed", 2027))

    # A deterministic, balanced discrete design: each option is repeated as
    # evenly as possible within every dimension, then dimensions are shuffled
    # independently.  This covers the supplied ranges without a multi-million
    # full Cartesian grid.
    columns: Dict[str, List[Dict[str, Any]]] = {}
    for index, (name, options) in enumerate(dimensions.items()):
        column = [copy.deepcopy(options[i % len(options)]) for i in range(count)]
        random.Random(seed + 1009 * (index + 1)).shuffle(column)
        columns[name] = column

    candidates: List[Dict[str, Any]] = [baseline]
    seen = {_signature(baseline)}
    for row in range(count * 8):
        parameters: Dict[str, Any] = {}
        for name, column in columns.items():
            option = column[row % count]
            if row >= count:
                option = random.Random(seed + row * 7919 + len(name)).choice(
                    dimensions[name]
                )
            parameters.update(option)
        parameters = _validate_parameters(parameters)
        signature = _signature(parameters)
        if signature not in seen:
            seen.add(signature)
            candidates.append(parameters)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise RuntimeError(f"Unable to generate {count} unique candidates")
    return candidates


@dataclass(frozen=True)
class Trial:
    candidate_id: str
    parameters: Dict[str, Any]
    seed: int
    config_path: Path
    output_dir: Path


def build_trials(manifest: Mapping[str, Any]) -> List[Trial]:
    generated_root = _resolve(
        str(manifest.get("generated_config_dir", "configs/stage_b/generated"))
    )
    output_root = _resolve(str(manifest.get("output_root", "log/stage_b/screen")))
    trials: List[Trial] = []
    for index, parameters in enumerate(build_candidates(manifest)):
        candidate_id = f"c{index:03d}"
        for seed_value in manifest.get("seeds", [42]):
            seed = int(seed_value)
            trials.append(
                Trial(
                    candidate_id=candidate_id,
                    parameters=parameters,
                    seed=seed,
                    config_path=generated_root
                    / candidate_id
                    / f"seed_{seed}.json",
                    output_dir=output_root / candidate_id / f"seed_{seed}",
                )
            )
    return trials


def _common_overrides(manifest: Mapping[str, Any], *, seed: int) -> Dict[str, Any]:
    overrides = dict(manifest.get("common_overrides", {}))
    task = str(manifest.get("task", "cifar10"))
    task_overrides = manifest.get("task_overrides", {})
    if isinstance(task_overrides, dict):
        selected = task_overrides.get(task, {})
        if not isinstance(selected, dict):
            raise ValueError(f"task_overrides.{task} must be an object")
        overrides.update(selected)
    overrides.update({"device": "cuda", "seed": seed})
    return overrides


def _pipeline_payload(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    output_dir: Path,
    defense: str,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides = _common_overrides(manifest, seed=seed)
    if parameters:
        overrides.update(parameters)
    attacks = manifest.get("attacks", ["none", "gn", "bd", "mix"])
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("attacks must be a non-empty array")
    return {
        "task": str(manifest.get("task", "cifar10")),
        "attacks": ",".join(str(item) for item in attacks),
        "defenses": defense,
        "log_dir": str(output_dir),
        "fed_config_file": str(
            manifest.get("fed_config_file", "configs/federated.json")
        ),
        "hyperparameters_file": str(
            manifest.get("hyperparameters_file", "configs/hyperparameters.json")
        ),
        "fed_config_overrides": overrides,
    }


def generate_configs(manifest: Mapping[str, Any]) -> List[Trial]:
    trials = build_trials(manifest)
    for trial in trials:
        _write_json(
            trial.config_path,
            _pipeline_payload(
                manifest,
                seed=trial.seed,
                output_dir=trial.output_dir,
                defense="svdd",
                parameters=trial.parameters,
            ),
        )
    if bool(manifest.get("run_fedavg_baseline", True)):
        generated_root = _resolve(
            str(manifest.get("generated_config_dir", "configs/stage_b/generated"))
        )
        output_root = _resolve(
            str(manifest.get("output_root", "log/stage_b/screen"))
        )
        for seed_value in manifest.get("seeds", [42]):
            seed = int(seed_value)
            _write_json(
                generated_root / "baseline" / f"seed_{seed}.json",
                _pipeline_payload(
                    manifest,
                    seed=seed,
                    output_dir=output_root / "baseline" / f"seed_{seed}",
                    defense="avg",
                ),
            )
    return trials


def _result_paths(
    manifest: Mapping[str, Any], output_dir: Path, defense: str
) -> List[Path]:
    task = str(manifest.get("task", "cifar10"))
    return [
        output_dir / f"{task}__{attack}__{defense}.json"
        for attack in manifest.get("attacks", [])
    ]


def _effective_matches(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, defense: str
) -> bool:
    meta = payload.get("meta", {})
    effective = meta.get("effective_config", {}) if isinstance(meta, dict) else {}
    if not isinstance(meta, dict) or not isinstance(effective, dict):
        return False
    if meta.get("defense") != defense:
        return False
    for key, value in expected.items():
        if key == "num_malicious":
            # The pipeline intentionally turns the no-attack control into an
            # all-benign population so its FPR labels match actual behavior.
            expected_malicious = 0 if meta.get("attack") == "none" else int(value)
            if int(meta.get("num_malicious", -1)) != expected_malicious:
                return False
        elif effective.get(key) != value:
            return False
    return True


def _config_complete(
    manifest: Mapping[str, Any], config_path: Path, output_dir: Path, defense: str
) -> bool:
    if not config_path.exists():
        return False
    try:
        config = _read_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = config.get("fed_config_overrides", {})
    expected_rounds = int(expected.get("total_rounds", -1))
    for path in _result_paths(manifest, output_dir, defense):
        if not path.exists():
            return False
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        rounds = payload.get("rounds", [])
        if not isinstance(rounds, list) or len(rounds) != expected_rounds:
            return False
        if not _effective_matches(payload, expected, defense=defense):
            return False
    return True


def _python_binary(manifest: Mapping[str, Any]) -> str:
    configured = str(manifest.get("python", "")).strip()
    if configured:
        path = _resolve(configured)
        return str(path) if path.exists() else configured
    return sys.executable


def run_trials(
    manifest: Mapping[str, Any],
    trials: Sequence[Trial],
    *,
    gpu: int,
    workers: int,
    force: bool,
) -> None:
    jobs: List[tuple[str, Path, Path, str]] = []
    generated_root = _resolve(
        str(manifest.get("generated_config_dir", "configs/stage_b/generated"))
    )
    output_root = _resolve(str(manifest.get("output_root", "log/stage_b/screen")))
    if bool(manifest.get("run_fedavg_baseline", True)):
        for seed_value in manifest.get("seeds", [42]):
            seed = int(seed_value)
            config_path = generated_root / "baseline" / f"seed_{seed}.json"
            output_dir = output_root / "baseline" / f"seed_{seed}"
            if force or not _config_complete(
                manifest, config_path, output_dir, "avg"
            ):
                jobs.append((f"baseline/seed_{seed}", config_path, output_dir, "avg"))
    for trial in trials:
        if force or not _config_complete(
            manifest, trial.config_path, trial.output_dir, "svdd"
        ):
            jobs.append(
                (
                    f"{trial.candidate_id}/seed_{trial.seed}",
                    trial.config_path,
                    trial.output_dir,
                    "svdd",
                )
            )
    print(
        f"SVDD calibration jobs={len(jobs)} workers={workers} gpu={gpu} "
        f"candidates={len(trials)}"
    )
    if not jobs:
        return

    work: queue.Queue[tuple[str, Path, Path, str]] = queue.Queue()
    for job in jobs:
        work.put(job)
    failures: List[tuple[str, int]] = []
    lock = threading.Lock()
    python = _python_binary(manifest)
    cpu_threads = str(int(manifest.get("cpu_threads_per_worker", 4)))

    def worker(worker_id: int) -> None:
        while True:
            try:
                name, config_path, output_dir, _defense = work.get_nowait()
            except queue.Empty:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            console_path = output_dir / "console.log"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "MKL_NUM_THREADS": cpu_threads,
                    "OMP_NUM_THREADS": cpu_threads,
                    "OPENBLAS_NUM_THREADS": cpu_threads,
                    "PYTHONUNBUFFERED": "1",
                }
            )
            with lock:
                print(f"[GPU {gpu}/W{worker_id}] START {name}", flush=True)
            with console_path.open("a", encoding="utf-8") as console:
                completed = subprocess.run(
                    [python, "-u", "-m", "src.pipeline", "--config", str(config_path)],
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                if completed.returncode == 0:
                    print(f"[GPU {gpu}/W{worker_id}] DONE  {name}", flush=True)
                else:
                    failures.append((name, completed.returncode))
                    print(
                        f"[GPU {gpu}/W{worker_id}] FAIL  {name} "
                        f"exit={completed.returncode} log={console_path}",
                        flush=True,
                    )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"SVDD calibration failures: {failures}")


def _round_summary(path: Path, last_n: int) -> Dict[str, float | None]:
    payload = _read_json(path)
    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"No rounds in {path}")
    selected = rounds[-last_n:]
    values: Dict[str, List[float]] = {
        "tacc": [],
        "tpr": [],
        "fpr": [],
        "dpr": [],
        "f1": [],
        "asr": [],
    }
    for item in selected:
        evaluation = item.get("evaluation", {}) if isinstance(item, dict) else {}
        if not isinstance(evaluation, dict):
            continue
        for source, target in (
            ("accuracy", "tacc"),
            ("tpr", "tpr"),
            ("fpr", "fpr"),
            ("dpr", "dpr"),
            ("backdoor_asr", "asr"),
        ):
            value = evaluation.get(source)
            if isinstance(value, (int, float)):
                values[target].append(float(value))
        precision = evaluation.get("dpr")
        recall = evaluation.get("tpr")
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            denominator = float(precision) + float(recall)
            values["f1"].append(
                0.0
                if denominator <= 0.0
                else 2.0 * float(precision) * float(recall) / denominator
            )
    return {
        key: statistics.mean(items) if items else None for key, items in values.items()
    }


def _mean(items: Iterable[float | None], default: float = 0.0) -> float:
    finite = [float(item) for item in items if item is not None]
    return statistics.mean(finite) if finite else default


def select_candidates(
    manifest: Mapping[str, Any], trials: Sequence[Trial]
) -> Dict[str, Any]:
    last_n = int(manifest.get("score_last_n_rounds", 10))
    attacks = [str(item) for item in manifest.get("attacks", [])]
    attack_conditions = [item for item in attacks if item != "none"]
    output_root = _resolve(str(manifest.get("output_root", "log/stage_b/screen")))
    baseline: Dict[tuple[int, str], Dict[str, float | None]] = {}
    for seed_value in manifest.get("seeds", [42]):
        seed = int(seed_value)
        for attack in attacks:
            baseline[(seed, attack)] = _round_summary(
                output_root
                / "baseline"
                / f"seed_{seed}"
                / f"{manifest['task']}__{attack}__avg.json",
                last_n,
            )

    settings = manifest.get("selection", {})
    if not isinstance(settings, dict):
        raise ValueError("selection must be an object")
    weights = settings.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError("selection.weights must be an object")
    max_clean_fpr = float(settings.get("max_clean_fpr", 0.10))
    max_attack_fpr = float(settings.get("max_mean_attack_fpr", 0.15))
    max_clean_drop = float(settings.get("max_clean_tacc_drop", 0.05))

    grouped: Dict[str, List[Trial]] = {}
    for trial in trials:
        grouped.setdefault(trial.candidate_id, []).append(trial)
    rankings: List[Dict[str, Any]] = []
    for candidate_id, candidate_trials in grouped.items():
        summaries: Dict[tuple[int, str], Dict[str, float | None]] = {}
        for trial in candidate_trials:
            for attack in attacks:
                summaries[(trial.seed, attack)] = _round_summary(
                    trial.output_dir
                    / f"{manifest['task']}__{attack}__svdd.json",
                    last_n,
                )
        clean_tacc = _mean(
            summaries[(trial.seed, "none")]["tacc"] for trial in candidate_trials
        )
        clean_fpr = _mean(
            summaries[(trial.seed, "none")]["fpr"] for trial in candidate_trials
        )
        baseline_clean = _mean(
            baseline[(trial.seed, "none")]["tacc"] for trial in candidate_trials
        )
        clean_drop = baseline_clean - clean_tacc
        attack_rows = [
            summaries[(trial.seed, attack)]
            for trial in candidate_trials
            for attack in attack_conditions
        ]
        mean_f1 = _mean(row["f1"] for row in attack_rows)
        worst_f1 = min(float(row["f1"] or 0.0) for row in attack_rows)
        mean_tpr = _mean(row["tpr"] for row in attack_rows)
        mean_attack_fpr = _mean(row["fpr"] for row in attack_rows)
        mean_tacc = _mean(row["tacc"] for row in attack_rows)
        tacc_gains = [
            float(summaries[(trial.seed, attack)]["tacc"] or 0.0)
            - float(baseline[(trial.seed, attack)]["tacc"] or 0.0)
            for trial in candidate_trials
            for attack in attack_conditions
        ]
        asr_reductions = [
            float(baseline[(trial.seed, attack)]["asr"])
            - float(summaries[(trial.seed, attack)]["asr"])
            for trial in candidate_trials
            for attack in attack_conditions
            if baseline[(trial.seed, attack)]["asr"] is not None
            and summaries[(trial.seed, attack)]["asr"] is not None
        ]
        mean_tacc_gain = _mean(tacc_gains)
        mean_asr_reduction = _mean(asr_reductions)
        eligible = (
            clean_fpr <= max_clean_fpr
            and mean_attack_fpr <= max_attack_fpr
            and clean_drop <= max_clean_drop
        )
        score = (
            float(weights.get("mean_f1", 0.40)) * mean_f1
            + float(weights.get("worst_f1", 0.20)) * worst_f1
            + float(weights.get("mean_tacc_gain", 0.15)) * mean_tacc_gain
            + float(weights.get("mean_asr_reduction", 0.10)) * mean_asr_reduction
            + float(weights.get("clean_tacc", 0.10)) * clean_tacc
            - float(weights.get("clean_fpr_penalty", 0.05)) * clean_fpr
        )
        rankings.append(
            {
                "candidate_id": candidate_id,
                "parameters": candidate_trials[0].parameters,
                "eligible": eligible,
                "score": score,
                "clean_tacc": clean_tacc,
                "clean_fpr": clean_fpr,
                "clean_tacc_drop_vs_fedavg": clean_drop,
                "mean_attack_tacc": mean_tacc,
                "mean_attack_tacc_gain_vs_fedavg": mean_tacc_gain,
                "mean_attack_tpr": mean_tpr,
                "mean_attack_fpr": mean_attack_fpr,
                "mean_attack_f1": mean_f1,
                "worst_attack_f1": worst_f1,
                "mean_backdoor_asr_reduction": mean_asr_reduction,
            }
        )
    rankings.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["score"]),
            -float(row["worst_attack_f1"]),
            float(row["clean_fpr"]),
            str(row["candidate_id"]),
        )
    )
    selected = rankings[0]["parameters"] if rankings else {}
    return {
        "protocol": "Stage-B AE-SVDD balanced discrete screening",
        "selection_uses": [
            "clean FedAvg fidelity",
            "attack F1/TPR/FPR",
            "TACC gain versus matched FedAvg",
            "backdoor ASR reduction when available",
        ],
        "last_n_rounds": last_n,
        "constraints": {
            "max_clean_fpr": max_clean_fpr,
            "max_mean_attack_fpr": max_attack_fpr,
            "max_clean_tacc_drop": max_clean_drop,
        },
        "selected_parameters": selected,
        "recommended_hyperparameters_patch": {
            "common": {key: selected[key] for key in COMMON_PARAMETER_KEYS},
            "defenses": {
                "svdd": {key: selected[key] for key in DEFENSE_PARAMETER_KEYS}
            },
        }
        if selected
        else {},
        "rankings": rankings,
    }


def promote_manifest(
    manifest: Mapping[str, Any], selection: Mapping[str, Any]
) -> Dict[str, Any]:
    defaults = manifest.get("promotion_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("promotion_defaults must be an object")
    top_k = int(defaults.get("top_k", 4))
    rankings = selection.get("rankings", [])
    if not isinstance(rankings, list) or len(rankings) < top_k:
        raise ValueError(f"Selection has fewer than {top_k} candidates")
    promoted = copy.deepcopy(dict(manifest))
    promoted["name"] = f"{manifest.get('name', 'stage_b')}_confirm"
    promoted.pop("num_candidates", None)
    promoted["explicit_candidates"] = [row["parameters"] for row in rankings[:top_k]]
    promoted["seeds"] = [int(seed) for seed in defaults.get("seeds", [42, 43, 44])]
    promoted["attacks"] = [
        str(attack)
        for attack in defaults.get(
            "attacks", ["none", "gn", "lf", "sf", "bd", "lie", "mix"]
        )
    ]
    promoted.setdefault("common_overrides", {})["total_rounds"] = int(
        defaults.get("total_rounds", 300)
    )
    promoted["generated_config_dir"] = str(
        defaults.get("generated_config_dir", "configs/stage_b/generated_confirm")
    )
    promoted["output_root"] = str(
        defaults.get("output_root", "log/stage_b/confirm")
    )
    promoted["selection_file"] = str(
        defaults.get("selection_file", "log/stage_b/confirm_selection.json")
    )
    return promoted


def load_manifest(path: str | Path) -> Dict[str, Any]:
    manifest = _read_json(path)
    build_candidates(manifest)
    attacks = manifest.get("attacks", [])
    if "none" not in attacks:
        raise ValueError("Stage-B selection requires the no-attack condition")
    if not bool(manifest.get("run_fedavg_baseline", True)):
        raise ValueError("Stage-B selection requires run_fedavg_baseline=true")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="AE-SVDD Stage-B calibration")
    parser.add_argument("--manifest", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("generate")
    run = commands.add_parser("run")
    run.add_argument("--gpu", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true")
    commands.add_parser("select")
    promote = commands.add_parser("promote")
    promote.add_argument("--selection")
    promote.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    trials = generate_configs(manifest)
    if args.command == "plan":
        print(
            f"task={manifest['task']} candidates={len(build_candidates(manifest))} "
            f"seeds={len(manifest.get('seeds', []))} "
            f"attacks={len(manifest['attacks'])} "
            f"pipeline_jobs={len(trials) + len(manifest.get('seeds', []))}"
        )
        for index, parameters in enumerate(build_candidates(manifest)):
            digest = hashlib.sha1(_signature(parameters).encode()).hexdigest()[:8]
            print(f"c{index:03d} {digest} {json.dumps(parameters, sort_keys=True)}")
        return
    if args.command == "generate":
        print(f"Generated {len(trials)} candidate pipeline configs")
        return
    if args.command == "run":
        selected = trials[: args.limit] if args.limit else trials
        run_trials(
            manifest,
            selected,
            gpu=int(args.gpu if args.gpu is not None else manifest.get("gpu", 0)),
            workers=int(
                args.workers
                if args.workers is not None
                else manifest.get("workers_per_gpu", 2)
            ),
            force=bool(args.force),
        )
        return
    if args.command == "select":
        selection = select_candidates(manifest, trials)
        output = str(manifest.get("selection_file", "log/stage_b/selection.json"))
        _write_json(output, selection)
        print(f"Saved Stage-B ranking: {_resolve(output)}")
        print(json.dumps(selection["selected_parameters"], sort_keys=True))
        return
    if args.command == "promote":
        selection = _read_json(
            args.selection
            or str(manifest.get("selection_file", "log/stage_b/selection.json"))
        )
        promoted = promote_manifest(manifest, selection)
        _write_json(args.output, promoted)
        print(f"Saved confirmation manifest: {_resolve(args.output)}")


if __name__ == "__main__":
    main()
