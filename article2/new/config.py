from dataclasses import dataclass


@dataclass
class FedConfig:
    """Federated learning and AE-SVDD hyperparameters."""

    # --- Federation ---
    num_clients: int = 10
    num_benign: int = 7
    total_rounds: int = 300
    attack_type: str = "sign_flipping"  # "gaussian_noise" | "label_flipping" | "sign_flipping" | custom

    # --- Client training ---
    client_lr: float = 0.1
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
    local_epochs: int = 1
    batch_size: int = 32

    # --- Attack params ---
    gaussian_sigma: float = 0.5
    sign_flip_scale: float = 1.0

    # --- AE / Encoder ---
    latent_dim: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-6
    ae_grad_clip: float = 1.0

    # --- Phase schedule ---
    phase1_rounds: int = 50
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

