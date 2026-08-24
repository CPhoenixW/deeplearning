#!/usr/bin/env python3
"""Launch the stable Fashion-MNIST AE-SVDD sensitivity-screen protocol.

The generic matrix runner is also used by unrelated task studies.  This small
entry point pins the Fashion-MNIST calibration and the fixed client-side
stability guards used by this experiment, without changing another study's
working tree overrides.
"""

from __future__ import annotations

from run_svdd_sensitivity_matrix import BASE_OVERRIDES, main


BASE_OVERRIDES.update(
    {
        "num_clients": 100,
        "total_rounds": 100,
        "client_lr": 0.1,
        "client_momentum": 0.9,
        "client_weight_decay": 0.0,
        "client_grad_clip": 5.0,
        "client_update_clip": 5.0,
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
        "svdd_input_dim": 4096,
        "svdd_normalization_eps": 1e-6,
        "center_ema_decay": 0.9,
        "svdd_grad_clip": 1.0,
        "center_init_quantile": 0.5,
        "phase2_recon_quantile": 0.8,
        "device": "cuda",
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
