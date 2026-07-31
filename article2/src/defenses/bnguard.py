from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from ..utils import build_svdd_feature_matrix, extract_bn_features, robust_scale_features, weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats


class BNGuardDefense(BaseDefense):
    """Lightweight BN-statistics OOD/backdoor detector (median + MAD distance)."""

    defense_name = "bnguard"

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        X = build_svdd_feature_matrix(client_state_dicts, extract_bn_features)
        X = robust_scale_features(X)

        ref = X.median(dim=0).values
        d = torch.norm(X - ref.unsqueeze(0), dim=1)
        med_d = torch.median(d)
        mad_d = 1.4826 * torch.median((d - med_d).abs())
        mad_d = torch.clamp(mad_d, min=1e-6)
        tau = float(self.config.bnguard_tau)
        threshold = med_d + tau * mad_d

        m = (d <= threshold).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = m / (m.sum() + 1e-12)

        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        n_kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase="bnguard | BNGuard",
            show_detection=True,
            monitor_items=[
                ("Defense", "BNGuard"),
                ("Dist threshold", f"{float(threshold.item()):.4f}"),
                ("Kept clients", f"{n_kept}/{k}"),
            ],
            participant_metrics={"bn_distance": d.detach().cpu()},
        )


__all__ = ["BNGuardDefense"]
