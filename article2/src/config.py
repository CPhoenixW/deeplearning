"""Experiment hyperparameters.

Assume the process current working directory is the project root (the directory
that contains ``src/``, ``data/``, and ``log/``). Then ``data_root="data"`` and
``run_matrix --log-dir log`` resolve to those folders.
"""

from dataclasses import dataclass


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

    # --- Phase schedule ---
    phase1_rounds: int = 5
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
    data_root: str = "./data"
    # Strict non-IID partition (paper-style):
    # for each client k, sample class probabilities q^(k) ~ Dir(alpha * p),
    # where p is uniform prior over classes. Then assign a fixed number of
    # samples to each client according to q^(k).
    # - None: IID split
    # - smaller alpha: stronger heterogeneity
    # - larger alpha: closer to IID
    dirichlet_alpha: float | None = 5.0
    # Backward-compatible alias (deprecated). If dirichlet_alpha is None and
    # this field is set, tasks.py will use this value.
    dirichlet_noniid_beta: float | None = None


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
}


def normalize_attack_name(name: str) -> str:
    k = name.lower().strip()
    return ATTACK_ALIASES.get(k, k)


def normalize_defense_name(name: str) -> str:
    k = name.lower().strip()
    return DEFENSE_ALIASES.get(k, k)
