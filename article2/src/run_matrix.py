from __future__ import annotations
import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .clients import ATTACK_REGISTRY, mixed_attack_for_client
    from .config import (
        ATTACK_ALIASES,
        DEFENSE_ALIASES,
        FedConfig,
        load_matrix_run_config,
        normalize_attack_name,
        normalize_defense_name,
        project_root,
    )
    from .main import run_federated
    from .server import DEFENSE_REGISTRY
    from .tasks import TASK_REGISTRY, get_task
except ImportError:
    from clients import ATTACK_REGISTRY, mixed_attack_for_client
    from config import (
        ATTACK_ALIASES,
        DEFENSE_ALIASES,
        FedConfig,
        load_matrix_run_config,
        normalize_attack_name,
        normalize_defense_name,
        project_root,
    )
    from main import run_federated
    from server import DEFENSE_REGISTRY
    from tasks import TASK_REGISTRY, get_task


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_tasks_arg(value: str | None, default_task_name: str) -> List[str]:
    """``all`` → 全部任务；逗号分隔；``None`` → 仅默认任务名。"""
    if value is None or value.strip() == "":
        return [default_task_name]
    v = value.strip().lower()
    if v == "all":
        return sorted(TASK_REGISTRY.keys())
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    return names


def validate_choice(name: str, value: str, registry: Dict[str, object]) -> None:
    if value not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown {name} {value!r}. Available: {available}")


def validate_lists(
    tasks: Sequence[str],
    attacks: Sequence[str],
    defenses: Sequence[str],
) -> None:
    for t in tasks:
        validate_choice("task_name", t, TASK_REGISTRY)
    for a in attacks:
        validate_choice("attack_type", a, ATTACK_REGISTRY)
    for d in defenses:
        validate_choice("defense_type", d, DEFENSE_REGISTRY)


def apply_defense_to_config(cfg: FedConfig, defense: str) -> None:
    """与 ``main.resolve_defense_name(..., use_svdd=None)`` 一致：只看 ``defense_type``。"""
    d = normalize_defense_name(defense)
    cfg.defense_type = d
    if d != "svdd":
        cfg.aggregation_method = d


def run_one_combo(
    base_cfg: FedConfig,
    task_name: str,
    attack: str,
    defense: str,
    output_dir: Path,
    prepared_dataloaders=None,
) -> Path:
    attack_id = normalize_attack_name(attack)
    defense_id = normalize_defense_name(defense)
    cfg = copy.deepcopy(base_cfg)
    cfg.task_name = task_name
    cfg.attack_type = attack_id
    apply_defense_to_config(cfg, defense_id)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rounds = run_federated(
        cfg,
        use_svdd=None,
        collect_metrics=True,
        prepared_dataloaders=prepared_dataloaders,
    )
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    if rounds is None:
        rounds = []

    payload: Dict[str, object] = {
        "meta": {
            "task_name": cfg.task_name,
            "attack": attack_id,
            "defense": defense_id,
            "num_classes": cfg.num_classes,
            "started_at": started_at,
            "finished_at": finished_at,
            "num_clients": cfg.num_clients,
            "num_benign": cfg.num_benign,
            "total_rounds": cfg.total_rounds,
            "defense_type": cfg.defense_type,
            "aggregation_method": cfg.aggregation_method,
            "data_root": cfg.data_root,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_noniid_beta": cfg.dirichlet_noniid_beta,
            "seed": cfg.seed,
            "device": cfg.device,
            "use_amp": cfg.use_amp,
            "channels_last": cfg.channels_last,
            "cuda_aggregation": cfg.cuda_aggregation,
            "reuse_client_model": cfg.reuse_client_model,
            "skip_redundant_attack_training": cfg.skip_redundant_attack_training,
            "round_diagnostics": cfg.round_diagnostics,
            "phase1_selection": cfg.phase1_selection,
            "trimmed_mean_ratio": cfg.trimmed_mean_ratio,
            "trimmed_mean_num_byzantine": cfg.trimmed_mean_num_byzantine,
            "krum_num_byzantine": cfg.krum_num_byzantine,
            "multi_krum_num_selected": cfg.multi_krum_num_selected,
            "svdd_input_mode": cfg.svdd_input_mode,
            "svdd_feature_mode": cfg.svdd_feature_mode,
            "param_descriptor_dim": cfg.param_descriptor_dim,
            "param_descriptor_seed": cfg.param_descriptor_seed,
            "param_descriptor_device": cfg.param_descriptor_device,
            "mixed_attack_types": cfg.mixed_attack_types,
            "mixed_attack_assignments": (
                {str(cid): mixed_attack_for_client(cfg, cid) for cid in range(cfg.num_benign, cfg.num_clients)}
                if attack_id == "mix" else {}
            ),
            "dmc_warmup_rounds": cfg.dmc_warmup_rounds,
            "dmc_tau": cfg.dmc_tau,
            "dmc_ema_decay": cfg.dmc_ema_decay,
            "dmc_min_keep": cfg.dmc_min_keep,
            "dmc_norm_weight": cfg.dmc_norm_weight,
            "dmc_direction_weight": cfg.dmc_direction_weight,
            "dmc_sign_weight": cfg.dmc_sign_weight,
            "dmc_sparsity_weight": cfg.dmc_sparsity_weight,
            "dmc_temporal_weight": cfg.dmc_temporal_weight,
            "dmc_score_ema_decay": cfg.dmc_score_ema_decay,
        },
        "round_metrics": rounds,
    }

    filename = f"{task_name}__{attack_id}__{defense_id}.json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_list_options() -> None:
    def _aka_short_to_long(aliases: Dict[str, str]) -> Dict[str, list[str]]:
        out: Dict[str, list[str]] = {}
        for long_n, short_n in aliases.items():
            out.setdefault(short_n, []).append(long_n)
        return out

    atk_old = _aka_short_to_long(ATTACK_ALIASES)
    def_old = _aka_short_to_long(DEFENSE_ALIASES)

    print("Tasks (dataset + model, config.task_name):")
    for k in sorted(TASK_REGISTRY.keys()):
        print(f"  - {k}")
    print("Attacks (short id; long names still accepted):")
    for k in sorted(ATTACK_REGISTRY.keys()):
        olds = atk_old.get(k, [])
        extra = f"  (aka {', '.join(olds)})" if olds else ""
        print(f"  - {k}{extra}")
    print("Defenses (short id; long names still accepted):")
    for k in sorted(DEFENSE_REGISTRY.keys()):
        olds = def_old.get(k, [])
        extra = f"  (aka {', '.join(olds)})" if olds else ""
        print(f"  - {k}{extra}")


def main() -> None:
    default_cfg = FedConfig()
    parser = argparse.ArgumentParser(
        description="按 ``config.MatrixRunConfig`` / ``load_matrix_run_config`` 跑任务×攻击×防御矩阵。",
    )
    parser.add_argument("--list", action="store_true", help="列出合法 task / attack / defense 键后退出。")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 文件，键为 MatrixRunConfig 字段，用于覆盖 ``config.DEFAULT_MATRIX_RUN``。",
    )
    args = parser.parse_args()

    if args.list:
        print_list_options()
        sys.exit(0)

    mr = load_matrix_run_config(args.config)

    task_names = parse_tasks_arg(mr.task, default_cfg.task_name)

    if mr.attacks.strip().lower() == "all":
        attacks = sorted(ATTACK_REGISTRY.keys())
    else:
        attacks = [normalize_attack_name(x) for x in parse_list_arg(mr.attacks)]

    defenses = [normalize_defense_name(x) for x in parse_list_arg(mr.defenses)]
    validate_lists(task_names, attacks, defenses)

    log_dir = mr.log_dir if mr.log_dir is not None else str(project_root() / "log")
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FedConfig()
    cfg.total_rounds = mr.total_rounds
    cfg.num_clients = mr.num_clients
    cfg.num_benign = mr.num_benign
    if mr.data_root is not None:
        cfg.data_root = mr.data_root
    cfg.local_epochs = int(mr.local_epochs)
    if mr.num_workers is not None:
        cfg.num_workers = int(mr.num_workers)
    cfg.use_amp = bool(mr.use_amp)
    cfg.channels_last = bool(mr.channels_last)
    cfg.cuda_aggregation = bool(mr.cuda_aggregation)
    cfg.reuse_client_model = bool(mr.reuse_client_model)
    cfg.skip_redundant_attack_training = bool(mr.skip_redundant_attack_training)
    cfg.round_diagnostics = bool(mr.round_diagnostics)
    if mr.phase1_selection is not None:
        cfg.phase1_selection = str(mr.phase1_selection)
    if mr.svdd_input_mode is not None:
        cfg.svdd_input_mode = str(mr.svdd_input_mode)
    if mr.svdd_feature_mode is not None:
        cfg.svdd_feature_mode = str(mr.svdd_feature_mode)
    if mr.param_descriptor_dim is not None:
        cfg.param_descriptor_dim = int(mr.param_descriptor_dim)
    if mr.param_descriptor_seed is not None:
        cfg.param_descriptor_seed = int(mr.param_descriptor_seed)
    if mr.param_descriptor_device is not None:
        cfg.param_descriptor_device = str(mr.param_descriptor_device)
    for name in (
        "flgmm_warmup_rounds",
        "flgmm_control_l",
        "flgmm_em_iters",
        "flanders_window",
        "flanders_sampling",
        "flanders_maxiter",
        "flanders_alpha",
        "flanders_beta",
        "flanders_num_clients_to_keep",
        "dmc_warmup_rounds",
        "dmc_tau",
        "dmc_ema_decay",
        "dmc_min_keep",
        "dmc_norm_weight",
        "dmc_direction_weight",
        "dmc_sign_weight",
        "dmc_sparsity_weight",
        "dmc_temporal_weight",
        "dmc_score_ema_decay",
    ):
        value = getattr(mr, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if mr.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = None if mr.dirichlet_alpha < 0 else float(mr.dirichlet_alpha)
    cfg.seed = mr.seed
    cfg.device = mr.device
    if mr.trimmed_mean_num_byzantine is not None:
        cfg.trimmed_mean_num_byzantine = mr.trimmed_mean_num_byzantine

    print(f"Tasks: {task_names}")
    print(f"Attacks: {attacks}")
    print(f"Defenses: {defenses}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"data_root: {cfg.data_root}")
    print(f"dirichlet_alpha: {cfg.dirichlet_alpha}")

    total = len(task_names) * len(attacks) * len(defenses)
    idx = 0
    for task_name in task_names:
        data_cfg = copy.deepcopy(cfg)
        data_cfg.task_name = task_name
        task = get_task(data_cfg)
        data_cfg.num_classes = task.num_classes
        print(f"Preparing shared dataloaders for task={task_name} ...")
        prepared_dataloaders = task.build_dataloaders(data_cfg)
        for attack in attacks:
            for defense in defenses:
                idx += 1
                print(f"\n=== [{idx}/{total}] task={task_name} attack={attack} defense={defense} ===")
                out_path = run_one_combo(
                    cfg,
                    task_name,
                    attack,
                    defense,
                    output_dir,
                    prepared_dataloaders=prepared_dataloaders,
                )
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
