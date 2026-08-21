"""Experiment hyperparameters.

By default, ``data_root`` points at ``<project>/data`` where ``<project>`` is the
directory that contains ``src/`` (resolved from this file), not the process cwd.
Override with ``FedConfig.data_root`` or ``--data-root`` when needed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


def project_root() -> Path:
    """``article2/`` root: parent of the ``src/`` package."""

    return Path(__file__).resolve().parent.parent


def _default_data_root() -> str:
    return str(project_root() / "data")


@dataclass
class FedConfig:
    """Federated learning and AE-SVDD hyperparameters."""

    # --- Federation ---
    num_clients: int = 50
    num_benign: int = 35
    total_rounds: int = 300
    # Number of trusted, clean training samples withheld from all clients and
    # used only for validation-driven server decisions. The paper default is 50;
    # vary this field directly for validation-size sensitivity experiments.
    server_validation_size: int = 50
    defense_type: str = "svdd"
    aggregation_method: str = "avg"
    trimmed_mean_ratio: float = 0.2
    # Paper-style coordinate-wise trimmed mean uses explicit Byzantine upper bound b.
    # If None, code falls back to estimating b from client composition.
    trimmed_mean_num_byzantine: int | None = None
    krum_num_byzantine: int | None = None
    multi_krum_num_selected: int | None = None
    # --- Attack type (short IDs: none, gn, lf, sf, bd, lie, minmax, minsum, mix; aliases accepted) ---
    attack_type: str = "bd"
    # Comma-separated attack IDs assigned deterministically across malicious
    # clients for the mixed-attack experiment (e.g.
    # ``lf,bd,gn,sf,lie,minmax,minsum``).
    mixed_attack_types: str = "lf,bd,gn,sf,lie,minmax,minsum"

    # --- Client training ---
    client_lr: float = 0.1
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
    # ``inverse_frequency`` is used for imbalanced image tasks such as COVID-19.
    # The resolved weights are populated after the client split is built.
    class_weight_mode: str = "none"
    client_class_weights: list[float] | None = None
    # ``balanced`` samples local client batches with replacement so each class
    # contributes comparable gradient mass without changing the held-out test set.
    client_sampling_mode: str = "none"
    # Fixed numerical-stability guards.  ``client_grad_clip`` bounds every
    # local SGD gradient norm; ``client_update_clip`` bounds the complete
    # post-attack model delta uploaded by a participant.  ``None`` keeps the
    # historical behavior for configurations that do not opt in.
    client_grad_clip: float | None = None
    client_update_clip: float | None = None
    # 每轮通信前：客户端在本地数据上完整遍历训练多少次（epoch），再上传参与聚合。
    local_epochs: int = 1
    batch_size: int = 64
    num_workers: int = 0
    # CUDA execution knobs. Matrix runs expose the same switches independently;
    # AMP/channels-last are workload-dependent and remain opt-in by default.
    use_amp: bool = False
    channels_last: bool = False
    cuda_aggregation: bool = False
    reuse_client_model: bool = False
    skip_redundant_attack_training: bool = False
    # Number of benign clients trained together by the optional CUDA vmap
    # executor.  ``1`` preserves the serial implementation.
    client_batch_group_size: int = 1
    # Full update-vector diagnostics are useful interactively but expensive:
    # they repeatedly scan every client's complete model on CPU.
    round_diagnostics: bool = True

    # --- Attack params ---
    # Gaussian noise attack: per-tensor N(μ, (scale·σ)²) where μ=mean(W), σ=std(W) on global weights;
    # upload = μ + scale * σ * ε, ε~N(0,1) i.i.d. (moment-matched per layer, scale with gaussian_sigma).
    gaussian_sigma: float = 0.3
    sign_flip_scale: float = 1.0
    # LIE/ALIE attack params.  The default follows Baruch et al. exactly:
    # s=floor(N/2)+1-M; z=Phi^-1((N-M-s)/(N-M)).  An explicit override is
    # retained only for controlled attack-strength sensitivity studies.
    lie_z_override: float | None = None
    # Min-Max / Min-Sum perturbation vector from Shejwalkar & Houmansadr:
    # ``std`` is the reference-code default; ``sign`` and ``unit_vec`` are
    # retained as the paper's alternative constructions.
    distance_attack_deviation: str = "std"

    # --- Backdoor attack params ---
    backdoor_target_label: int = 0
    backdoor_poison_ratio: float = 0.6
    backdoor_trigger_size: int = 5       # square trigger size (in pixels)
    backdoor_trigger_value: float = 1.0  # pixel value in [0,1] since CIFAR transforms use ToTensor()
    # Backdoor model-replacement strength.
    # upload = global + scale * (local - global), scale=1.0 means no amplification.
    backdoor_model_replace_scale: float = 3.0

    # --- AE / Encoder ---
    latent_dim: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-6
    ae_grad_clip: float = 1.0
    # Input representation for the optional task-specific feature fallback.
    # The default fixed descriptor always describes client model deltas.
    svdd_input_mode: str = "delta"
    # Feature interface before AE-SVDD:
    # - "fixed_projection": fixed hierarchical multi-view Phi(delta W), default.
    # - "task": optional task-specific BN / LayerNorm fallback for old studies.
    svdd_feature_mode: str = "fixed_projection"
    param_descriptor_dim: int = 4096
    param_descriptor_seed: int = 2027
    # Fixed descriptor view allocation. Ratios must be non-negative and sum to 1.
    # Zero-valued views support controlled structural ablations.
    param_descriptor_global_ratio: float = 0.5
    param_descriptor_layer_ratio: float = 0.375
    param_descriptor_statistics_ratio: float = 0.125
    # "cpu" is deterministic; "cuda" is faster but scatter reductions can have
    # small floating-point ordering differences. "auto" follows the run device.
    param_descriptor_device: str = "cpu"

    # --- Phase schedule ---
    phase1_rounds: int = 15

    # --- SOTA defenses (ported from experiment/FL-Byzantine-Library) ---
    # LASA: Layer-Adaptive Sparsified model Aggregation (WACV 2025)
    lasa_sparsity_ratio: float = 0.9
    lasa_lambda_n: float = 1.0
    lasa_lambda_s: float = 1.0
    # FedSECA: Sign Election + Coordinate-wise Aggregation (CVPR 2025)
    # γ in the paper/code: fraction to zero out (keep top (1-γ) coords).
    fedseca_sparsity_gamma: float = 0.9
    fedseca_temperature: float = 1.0
    # FL-Defender: PCA-on-cosine-similarity + reputation accumulation
    # (ported without sklearn dependency; math-equivalent PCA via SVD).
    fldefender_pca_components: int = 2
    fldefender_q1: float = 0.25
    # AlignIns: TDA (direction) + MPSA (principal-sign on top-|Δ| coords)
    alignins_sparsity: float = 0.9
    alignins_lambda_s: float = 1.0
    alignins_lambda_c: float = 1.0
    # BNGuard: BN-feature L2 distance from median + tau * MAD
    bnguard_tau: float = 3.0
    # FLGMM: 1-D GMM over local-vs-temporary-global distances + SPC upper limit.
    flgmm_warmup_rounds: int = 50
    flgmm_control_l: float = 3.0
    flgmm_em_iters: int = 50
    # FLANDERS: Matrix autoregressive prediction over sampled parameter time series.
    flanders_window: int = 5
    flanders_sampling: int = 500
    flanders_maxiter: int = 100
    flanders_alpha: float = 1.0
    flanders_beta: float = 1.0
    flanders_num_clients_to_keep: int | None = None
    # FedDMC-style data-free multi-view malicious-client detector.  The
    # detector fuses magnitude, direction, sign, sparsity and temporal
    # consistency instead of assuming a particular attack family.
    dmc_warmup_rounds: int = 3
    dmc_tau: float = 3.0
    dmc_ema_decay: float = 0.8
    dmc_min_keep: int = 1
    dmc_norm_weight: float = 1.0
    dmc_direction_weight: float = 1.0
    dmc_sign_weight: float = 1.0
    dmc_sparsity_weight: float = 0.5
    dmc_temporal_weight: float = 1.0
    dmc_score_ema_decay: float = 0.7

    # --- SVDD ---
    center_ema_decay: float = 0.9
    svdd_grad_clip: float = 1.0
    # Single tunable SVDD loss-mixing coefficient.  svdd_lambda=1 is pure
    # SVDD; svdd_lambda=0 is pure reconstruction loss.  This coefficient is
    # independent of the Dirichlet alpha and of the client-selection score.
    svdd_lambda: float = 0.5
    # New protocol fields.  ``None`` lets the compatibility field below
    # resolve historical configs; formal experiments set Phase 2 explicitly.
    phase1_score_mode: str | None = None
    phase2_score_mode: str | None = None
    # Deprecated compatibility field.  ``legacy`` means reconstruction in
    # Phase 1 and SVDD distance in Phase 2.  A non-legacy value preserves old
    # sensitivity runs that applied one mode to both phases.
    svdd_score_mode: str = "legacy"
    # Trusted-sample quantiles used by center initialization and Phase-2
    # reconstruction training. Values must be in (0, 1].
    center_init_quantile: float = 0.5
    phase2_recon_quantile: float = 0.8
    svdd_feature_clip: float = 10.0

    # --- Task (dataset + backbone) ---
    # task_name keys must exist in tasks.TASK_REGISTRY, e.g. "cifar10", "covid19", "ag_news"
    task_name: str = "cifar10"
    # Optional AG News feature fallback, used only when svdd_feature_mode="task".
    # The default fixed descriptor does not read this field.
    ag_news_svdd_features: str = "ln_bn"
    # Set automatically from the task in main.run_federated; used by label-flip etc.
    num_classes: int = 10

    # --- Misc ---
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # CIFAR-10: torchvision uses <data_root>/cifar-10-batches-py/ OR <data_root>/cifar10/cifar-10-batches-py/
    # (auto-detected). Fashion-MNIST uses <data_root>/fashion_mnist/FashionMNIST/.
    # COVID-19 Radiography uses <data_root>/covid19/COVID-19_Radiography_Dataset/.
    # AG News uses <data_root>/ag_news/hf_cache (HuggingFace datasets).
    data_root: str = field(default_factory=_default_data_root)
    # Keep existing offline behavior by default. Fresh machines can set this
    # to false in JSON so AG News is downloaded into data_root on first use.
    hf_datasets_offline: bool = True
    # Strict non-IID partition (paper-style):
    # for each client k, sample class probabilities q^(k) ~ Dir(alpha * p),
    # where p is uniform prior over classes. Then assign a fixed number of
    # samples to each client according to q^(k).
    # - None: IID split
    # - smaller alpha: stronger heterogeneity
    # - larger alpha: closer to IID
    dirichlet_alpha: float | None = 1
    # Backward-compatible alias (deprecated). If dirichlet_alpha is None and
    # this field is set, tasks.py will use this value.
    dirichlet_noniid_beta: float | None = None


@dataclass
class PipelineConfig:
    """JSON-driven task × attack × defense pipeline configuration."""

    task: str = "cifar10"
    attacks: str = "all"
    defenses: str = "all"
    log_dir: str | None = None
    fed_config_file: str | None = None
    hyperparameters_file: str | None = None
    fed_config_overrides: dict[str, object] = field(default_factory=dict)


DEFAULT_PIPELINE_RUN = PipelineConfig()


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """Load one validated JSON pipeline configuration."""

    cfg = copy.deepcopy(DEFAULT_PIPELINE_RUN)
    if path is None:
        return cfg
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline config JSON must be a single object at the top level.")
    for key, value in data.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown PipelineConfig field: {key!r}")
        setattr(cfg, key, value)
    return cfg


# --- Attack / defense short names (canonical); long CLI names map here ---
ATTACK_ALIASES: dict[str, str] = {
    "none": "none",
    "no_attack": "none",
    "mix": "mix",
    "mixed": "mix",
    "hybrid": "mix",
    "gaussian_noise": "gn",
    "label_flipping": "lf",
    "sign_flipping": "sf",
    "backdoor": "bd",
    "lie_attack": "lie",
    "alie": "lie",
    "minmax": "minmax",
    "min_max": "minmax",
    "min-max": "minmax",
    "minmax_attack": "minmax",
    "minsum": "minsum",
    "min_sum": "minsum",
    "min-sum": "minsum",
    "minsum_attack": "minsum",
}

DEFENSE_ALIASES: dict[str, str] = {
    "fedavg": "avg",
    "trimmed_mean": "tm",
    "multi_krum": "mk",
    "fedseca": "seca",
    "fl_defender": "fld",
    "align_ins": "alignins",
    "bn_guard": "bnguard",
    "fl_gmm": "flgmm",
    "feddmc": "dmc",
    "fed_dmc": "dmc",
    "fed-dmc": "dmc",
    "multi_view": "dmc",
    "multiview": "dmc",
}


def normalize_attack_name(name: str) -> str:
    k = name.lower().strip()
    return ATTACK_ALIASES.get(k, k)


def normalize_defense_name(name: str) -> str:
    k = name.lower().strip()
    return DEFENSE_ALIASES.get(k, k)


def resolve_fed_config_path(path: str | Path | None = None) -> Path:
    """Resolve the modular pipeline's federated config without cwd dependence."""

    return (project_root() / "configs" / "federated.json") if path is None else (
        project_root() / path if not Path(path).is_absolute() else Path(path)
    )


def resolve_hyperparameters_path(path: str | Path | None = None) -> Path:
    """Resolve the modular pipeline's hyperparameter table."""

    return (project_root() / "configs" / "hyperparameters.json") if path is None else (
        project_root() / path if not Path(path).is_absolute() else Path(path)
    )


def load_fed_config_values(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Federated config must be a JSON object.")
    values = payload.get("values", payload)
    if not isinstance(values, dict):
        raise ValueError("Federated config 'values' must be a JSON object.")
    return dict(values)


def apply_fed_config_overrides(
    config: FedConfig,
    values: dict[str, object] | None,
    *,
    source: str = "overrides",
) -> FedConfig:
    """Apply validated JSON values to ``FedConfig``.

    The server config historically used ``num_malicious`` while Python uses
    ``num_benign``.  Both are accepted so old experiment files remain usable.
    """

    if not values:
        return config
    valid = {item.name for item in fields(config)}
    pending_malicious = values.get("num_malicious")
    for key, value in values.items():
        if key == "num_malicious":
            continue
        # ``alpha`` was historically used for the SVDD loss coefficient.
        # Keep old generated experiment configs readable while exposing the
        # unambiguous ``svdd_lambda`` field in new configs and results.
        if key == "alpha":
            if "svdd_lambda" not in values:
                setattr(config, "svdd_lambda", value)
            continue
        if key not in valid:
            raise ValueError(f"Unknown FedConfig field {key!r} in {source}.")
        if key in {"data_root"} and isinstance(value, str) and not Path(value).is_absolute():
            value = str(project_root() / value)
        setattr(config, key, value)
    if pending_malicious is not None:
        config.num_benign = int(config.num_clients) - int(pending_malicious)
    if "mixed_attack_types" in values and values["mixed_attack_types"] is None:
        config.mixed_attack_types = "lf,bd,gn,sf,lie,minmax,minsum"
    return config


def load_hyperparameter_table(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hyperparameter table must be a JSON object.")
    return payload


def resolve_hyperparameters(
    table: dict[str, object], attack: str, defense: str, task: str
) -> dict[str, object]:
    """Merge common, attack-profile, defense and task-specific overrides."""

    result: dict[str, object] = {}
    common = table.get("common", {})
    if isinstance(common, dict):
        result.update(common)
    profiles = table.get("profiles", {})
    profile = profiles.get(attack, {}) if isinstance(profiles, dict) else {}
    if isinstance(profile, dict):
        values = profile.get("common", {})
        if isinstance(values, dict):
            result.update(values)
        profile_defenses = profile.get("defenses", {})
        if isinstance(profile_defenses, dict) and isinstance(profile_defenses.get(defense), dict):
            result.update(profile_defenses[defense])
    defenses = table.get("defenses", {})
    if isinstance(defenses, dict) and isinstance(defenses.get(defense), dict):
        result.update(defenses[defense])
    tasks = table.get("tasks", {})
    if isinstance(tasks, dict) and isinstance(tasks.get(task), dict):
        result.update(tasks[task])
    return result
