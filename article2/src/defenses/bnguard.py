from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from ..utils import build_svdd_feature_matrix, extract_bn_features, robust_scale_features, weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats


class BNGuardDefense(BaseDefense):
    """Lightweight BN-statistics OOD/backdoor detector (median + MAD distance).

    Some task backbones (notably the lightweight Fashion-MNIST CNN) do not
    contain BatchNorm layers.  In that case we retain the same median/MAD
    decision rule but build a compact per-layer parameter-delta feature vector
    (norm, mean absolute value, standard deviation, maximum absolute value, and
    nonzero ratio).  The fallback is explicit in the round monitor so it is not
    mistaken for genuine BN-statistics evidence.
    """

    defense_name = "bnguard"

    def _parameter_delta_features(
        self,
        client_state: Dict[str, Tensor],
        global_state: Dict[str, Tensor],
    ) -> Tensor:
        """Return stable low-dimensional features when a model has no BN stats."""

        features: List[Tensor] = []
        for name in self.param_names:
            value = client_state[name].detach().cpu().float().reshape(-1)
            reference = global_state[name].detach().cpu().float().reshape(-1)
            delta = torch.nan_to_num(
                value - reference,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if delta.numel() == 0:
                features.extend([torch.tensor(0.0)] * 5)
                continue
            features.extend(
                [
                    delta.norm(p=2),
                    delta.abs().mean(),
                    delta.std(unbiased=False),
                    delta.abs().max(),
                    (delta.abs() > 1e-12).float().mean(),
                ]
            )
        if not features:
            return torch.zeros(1, dtype=torch.float32)
        return torch.stack(features).float()

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        feature_source = "BN statistics"
        try:
            X = build_svdd_feature_matrix(client_state_dicts, extract_bn_features)
        except ValueError as exc:
            if "No BatchNorm statistics" not in str(exc):
                raise
            feature_source = "parameter-delta fallback"
            global_sd = self.state_dict_for_clients()
            X = torch.stack(
                [self._parameter_delta_features(sd, global_sd) for sd in client_state_dicts],
                dim=0,
            )
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
                ("Feature source", feature_source),
                ("Dist threshold", f"{float(threshold.item()):.4f}"),
                ("Kept clients", f"{n_kept}/{k}"),
            ],
            participant_metrics={"bn_distance": d.detach().cpu()},
        )


__all__ = ["BNGuardDefense"]
