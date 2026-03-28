from dataclasses import dataclass


@dataclass
class FedConfig:
    """Federated learning and AE-SVDD hyperparameters."""

    # --- Federation ---
    num_clients: int = 50
    num_benign: int = 35
    total_rounds: int = 300
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
    backdoor_poison_ratio: float = 0.2  # fraction of samples per batch to poison
    backdoor_trigger_size: int = 3       # square trigger size (in pixels)
    backdoor_trigger_value: float = 1.0  # pixel value in [0,1] since CIFAR transforms use ToTensor()

    # --- AE / Encoder ---
    latent_dim: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-6
    ae_grad_clip: float = 1.0

    # --- Phase schedule ---
    phase1_rounds: int = 10
    buffer_capacity: int = 500

    # --- SVDD ---
    svdd_warmup_rounds: int = 100
    center_ema_decay: float = 0.9
    tau_multiplier: float = 3.0
    softweight_T_start: float = 5.0
    softweight_T_end: float = 0.5
    svdd_grad_clip: float = 1.0
    svdd_recon_lambda: float = 0.1

    # --- Misc ---
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    data_root: str = "./data"

