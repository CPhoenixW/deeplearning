"""Experiment hyperparameters.

By default, ``data_root`` points at ``<project>/data`` where ``<project>`` is the
directory that contains ``src/`` (resolved from this file), not the process cwd.
Override with ``FedConfig.data_root`` or ``--data-root`` when needed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
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
    defense_type: str = "svdd"
    aggregation_method: str = "avg"
    trimmed_mean_ratio: float = 0.2
    # Paper-style coordinate-wise trimmed mean uses explicit Byzantine upper bound b.
    # If None, code falls back to estimating b from client composition.
    trimmed_mean_num_byzantine: int | None = None
    krum_num_byzantine: int | None = None
    multi_krum_num_selected: int | None = None
    # --- Attack type (short IDs: gn, lf, sf, bd, lie; long names still accepted) ---
    attack_type: str = "bd"

    # --- Client training ---
    client_lr: float = 0.1
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
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
    # Full update-vector diagnostics are useful interactively but expensive:
    # they repeatedly scan every client's complete model on CPU.
    round_diagnostics: bool = True

    # --- Attack params ---
    # Gaussian noise attack: per-tensor N(μ, (scale·σ)²) where μ=mean(W), σ=std(W) on global weights;
    # upload = μ + scale * σ * ε, ε~N(0,1) i.i.d. (moment-matched per layer, scale with gaussian_sigma).
    gaussian_sigma: float = 0.3
    sign_flip_scale: float = 1.0
    # LIE/ALIE attack params:
    # malicious update delta = mu + z * sigma, where mu/sigma are estimated from
    # benign-client deltas in the current round. z defaults to z_max satisfying
    # Phi(z) < (N - M - s) / (N - M), with N total clients, M malicious clients.
    lie_s: int | None = None
    lie_z_override: float | None = None

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
    # Input representation used by AE-SVDD before robust feature scaling:
    # - "absolute": task-specific features extracted from each uploaded client model.
    # - "delta": the same features extracted from (client model - pre-round global model).
    #
    # With the current per-round coordinate-wise median/MAD centering, these two
    # modes are translation-equivalent for linear parameter extractors.  The option
    # is retained explicitly so that this equivalence can be tested and so future
    # scaling strategies can compare the two representations without code changes.
    svdd_input_mode: str = "absolute"
    # Feature interface before AE-SVDD:
    # - "task": existing task-specific BN / LayerNorm descriptor.
    # - "fixed_projection": fixed hierarchical multi-view Phi(delta W).
    svdd_feature_mode: str = "task"
    param_descriptor_dim: int = 4096
    param_descriptor_seed: int = 2027
    # "cpu" is deterministic; "cuda" is faster but scatter reductions can have
    # small floating-point ordering differences. "auto" follows the run device.
    param_descriptor_device: str = "cpu"

    # --- Phase schedule ---
    phase1_rounds: int = 15
    # AE warm-up: in robust-scaled BN space, keep clients closest to the coordinate-wise
    # median; only they contribute to the AE backward step and this round's FedAvg.
    ae_warmup_keep_ratio: float = 0.8

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

    # --- SVDD ---
    svdd_warmup_rounds: int = 100
    center_ema_decay: float = 0.9
    # Threshold schedule for SVDD filtering:
    # threshold = median(d) + tau * MAD(d)
    # tau anneals linearly from tau_start to tau_end in Phase 2.
    tau_start: float = 3.0
    tau_end: float = 2.0
    # Backward-compatible fixed tau; only used when tau_start/tau_end are invalid.
    tau_multiplier: float = 3.0
    svdd_grad_clip: float = 1.0
    svdd_recon_lambda: float = 0.1

    # --- Task (dataset + backbone) ---
    # task_name keys must exist in tasks.TASK_REGISTRY, e.g. "cifar10", "fashion_mnist", "ag_news"
    task_name: str = "cifar10"
    # ag_news + SVDD only: "ln" = Transformer LayerNorm γ/β; "bn" = BN head only;
    # "ln_bn" = concat LN+BN (~2048-D vs ~1024-D BN-only) for stronger detection signal.
    ag_news_svdd_features: str = "ln_bn"
    # Set automatically from the task in main.run_federated; used by label-flip etc.
    num_classes: int = 10

    # --- Misc ---
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # CIFAR-10: torchvision uses <data_root>/cifar-10-batches-py/ OR <data_root>/cifar10/cifar-10-batches-py/
    # (auto-detected). Fashion-MNIST uses <data_root>/fashion_mnist/FashionMNIST/.
    # AG News uses <data_root>/ag_news/hf_cache (HuggingFace datasets).
    data_root: str = field(default_factory=_default_data_root)
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
class MatrixRunConfig:
    """``run_matrix`` 默认：任务 × 攻击 × 防御网格；字段可被 JSON（``--config``）覆盖。"""

    # 与 argparse 时期一致：task 为 ``all``、逗号分隔或单个名；attacks/defenses 为 ``all`` 或逗号分隔。
    task: str = "cifar10"
    attacks: str = "all"
    defenses: str = "all"
    # ``None`` → ``<project>/log``
    log_dir: str | None = None
    total_rounds: int = 300
    num_clients: int = 50
    # 矩阵脚本历史上默认 40，与 ``FedConfig`` 默认 35 区分。
    num_benign: int = 40
    data_root: str | None = None
    local_epochs: int = 1
    num_workers: int | None = None
    use_amp: bool = False
    channels_last: bool = False
    cuda_aggregation: bool = True
    reuse_client_model: bool = True
    skip_redundant_attack_training: bool = True
    round_diagnostics: bool = False
    svdd_input_mode: str | None = None
    svdd_feature_mode: str | None = None
    param_descriptor_dim: int | None = None
    param_descriptor_seed: int | None = None
    param_descriptor_device: str | None = None
    flgmm_warmup_rounds: int | None = None
    flgmm_control_l: float | None = None
    flgmm_em_iters: int | None = None
    flanders_window: int | None = None
    flanders_sampling: int | None = None
    flanders_maxiter: int | None = None
    flanders_alpha: float | None = None
    flanders_beta: float | None = None
    flanders_num_clients_to_keep: int | None = None
    # ``None`` 表示不覆盖 ``FedConfig`` 默认（例如 dirichlet）。
    dirichlet_alpha: float | None = None
    seed: int = 42
    device: str = "cuda"
    trimmed_mean_num_byzantine: int | None = None


DEFAULT_MATRIX_RUN = MatrixRunConfig()


def load_matrix_run_config(path: str | Path | None = None) -> MatrixRunConfig:
    """深拷贝 ``DEFAULT_MATRIX_RUN``，若给定 ``path`` 则按 JSON 键合并（未知键报错）。"""

    cfg = copy.deepcopy(DEFAULT_MATRIX_RUN)
    if path is None:
        return cfg
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Matrix run config JSON must be a single object at the top level.")
    for key, value in data.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown MatrixRunConfig field: {key!r}")
        setattr(cfg, key, value)
    return cfg


# --- Attack / defense short names (canonical); long CLI names map here ---
ATTACK_ALIASES: dict[str, str] = {
    "gaussian_noise": "gn",
    "label_flipping": "lf",
    "sign_flipping": "sf",
    "backdoor": "bd",
    "lie_attack": "lie",
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
}


def normalize_attack_name(name: str) -> str:
    k = name.lower().strip()
    return ATTACK_ALIASES.get(k, k)


def normalize_defense_name(name: str) -> str:
    k = name.lower().strip()
    return DEFENSE_ALIASES.get(k, k)
