from dataclasses import dataclass


@dataclass
class FedConfig:
    """Federated learning and AE-SVDD hyperparameters."""

    # --- Federation ---
    num_clients: int = 50
    num_benign: int = 35
    total_rounds: int = 300
    # Unified defense selector:
    # - "svdd"
    # - "fedavg"
    # - "trimmed_mean"
    # - "multi_krum"
    defense_type: str = "svdd"
    # Aggregation / defense interface (used when use_svdd=False in main.py):
    # - "fedavg"
    # - "trimmed_mean"
    # - "multi_krum"
    aggregation_method: str = "fedavg"
    # Trimmed Mean ratio in [0, 0.5). 0.2 means trim 20% smallest/largest updates.
    trimmed_mean_ratio: float = 0.2
    # Optional byzantine client estimate for Multi-Krum. If None, use (num_clients - num_benign).
    krum_num_byzantine: int | None = None
    # Number of selected updates in Multi-Krum averaging. If None, use (num_clients - f - 2).
    multi_krum_num_selected: int | None = None
    # attack_type:
    # - "gaussian_noise"
    # - "label_flipping"
    # - "sign_flipping"
    # - "backdoor"        (trigger + target label)
    # - "lie_attack"      (ALIE/LIE-inspired stealthy update)
    # - custom
    attack_type: str = "gaussian_noise"

    # --- Client training ---
    client_lr: float = 0.1
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
    local_epochs: int = 1
    batch_size: int = 32

    # --- Attack params ---
    gaussian_sigma: float = 0.5
    sign_flip_scale: float = 1.0

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
    phase1_rounds: int = 15
    buffer_capacity: int = 500

    # --- SVDD ---
    svdd_warmup_rounds: int = 20
    center_ema_decay: float = 0.9
    # Threshold schedule for SVDD filtering:
    # threshold = median(d) + tau * MAD(d)
    # tau anneals linearly from tau_start to tau_end in Phase 2.
    tau_start: float = 2.0
    tau_end: float = 0
    # Backward-compatible fixed tau; only used when tau_start/tau_end are invalid.
    tau_multiplier: float = 3.0
    softweight_T_start: float = 5.0
    softweight_T_end: float = 0.5
    svdd_grad_clip: float = 1.0
    svdd_recon_lambda: float = 0.1

    # --- Task (dataset + backbone) ---
    # task_name keys must exist in tasks.TASK_REGISTRY, e.g. "cifar10", "fashion_mnist"
    task_name: str = "cifar10"
    # Set automatically from the task in main.run_federated; used by label-flip etc.
    num_classes: int = 10

    # --- Misc ---
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    data_root: str = "./data"

