from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .base import BaseDefense, DefenseResult as RoundStats
from .common import _apply_flat_delta_to_global, _flatten_delta, _mad_zscores


class AlignInsDefense(BaseDefense):
    """AlignIns-style update filtering (TDA + MPSA) on flattened floating deltas.

    Ported conservatively from experiment/AlignIns/src/aggregation.py:
    - TDA: cosine alignment of each client delta with the mean update direction.
    - MPSA: sign agreement with the majority sign on top-|Δ| coordinates.
    Outliers are flagged via MAD-z thresholds; kept clients are norm-clipped and averaged.
    """

    defense_name = "alignins"

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_delta(global_sd, sd) for sd in client_state_dicts], dim=0
        )  # (K, D)

        agg_dir = deltas.mean(dim=0)
        agg_norm = agg_dir.norm().clamp(min=1e-12)
        delta_norms = deltas.norm(dim=1).clamp(min=1e-12)
        tda = (deltas @ agg_dir) / (delta_norms * agg_norm)

        major_sign = torch.sign(deltas.sum(dim=0))
        sparsity = float(self.config.alignins_sparsity)
        sparsity = min(max(sparsity, 1e-6), 1.0)
        k_dim = max(1, int(sparsity * int(deltas.shape[1])))
        k_dim = min(k_dim, int(deltas.shape[1]))

        mpsa = torch.empty(k, dtype=torch.float32)
        for i in range(k):
            _, idx = torch.topk(deltas[i].abs(), k_dim)
            agree = (torch.sign(deltas[i, idx]) == major_sign[idx]).float().mean()
            mpsa[i] = agree

        mz_mpsa = _mad_zscores(mpsa)
        mz_tda = _mad_zscores(tda.float())
        lambda_s = float(self.config.alignins_lambda_s)
        lambda_c = float(self.config.alignins_lambda_c)
        m = ((mz_mpsa < lambda_s) & (mz_tda < lambda_c)).float()
        if m.sum() < 1:
            m = torch.ones_like(m)

        d = torch.max(mz_mpsa, mz_tda)
        alpha = m / (m.sum() + 1e-12)

        kept_idx = (m > 0.5).nonzero(as_tuple=False).view(-1)
        kept_updates = deltas[kept_idx]
        updates_norm = kept_updates.norm(dim=1, keepdim=True)
        norm_clip = updates_norm.median(dim=0).values.clamp(min=1e-12)
        clipped_norm = torch.clamp(updates_norm, max=norm_clip)
        kept_updates = kept_updates / updates_norm * clipped_norm
        delta_agg = kept_updates.mean(dim=0)

        new_global_sd = _apply_flat_delta_to_global(global_sd, delta_agg)
        self.global_model.load_state_dict(new_global_sd)

        n_kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase="alignins | AlignIns",
            show_detection=True,
            monitor_items=[
                ("Defense", "AlignIns"),
                ("TDA (avg cos)", f"{float(tda.mean().item()):.4f}"),
                ("MPSA (avg)", f"{float(mpsa.mean().item()):.4f}"),
                ("Kept clients", f"{n_kept}/{k}"),
            ],
            participant_metrics={
                "anomaly_score": d.detach().cpu(),
                "tda": tda.detach().cpu(),
                "mpsa": mpsa.detach().cpu(),
            },
        )


__all__ = ["AlignInsDefense"]
