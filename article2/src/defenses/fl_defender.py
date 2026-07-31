from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from ..config import FedConfig
from .base import BaseDefense, DefenseResult as RoundStats
from .common import (
    _aggregate_nonparam_buffers,
    _apply_flat_param_delta_to_global,
    _cosine_similarity_matrix,
    _flatten_param_delta,
    _pca_2d_via_svd,
    _standardize_matrix,
)


class FLDefenderDefense(BaseDefense):
    """Port of FL-Defender from experiment/FL-Byzantine-Library/aggregators/fl_defender.py.

    Keeps the same steps but avoids sklearn dependency (standardization + PCA via SVD).
    """

    defense_name = "fld"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self.n_clients = int(config.num_clients)
        self.score_history = np.zeros(self.n_clients, dtype=np.float64)
        self.rounds = 0

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        )  # (K,D)

        grads_np = deltas.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size == 0:
                delta_agg = torch.mean(deltas, dim=0)
                new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
                self.global_model.load_state_dict(new_global_sd)
                m = torch.ones(k)
                alpha = torch.full((k,), 1.0 / max(1, k))
                return RoundStats(
                    center_norm=float("nan"),
                    z_var=0.0,
                    ae_loss=float("nan"),
                    svdd_loss=float("nan"),
                    d=torch.zeros(k),
                    m=m,
                    alpha=alpha,
                    phase="fld | FL-Defender (fallback)",
                    show_detection=False,
                    monitor_items=[("Defense", "FL-Defender"), ("Invalid grads", f"{n_invalid}/{k}")],
                )
            grads_np_valid = grads_np[valid_indices]
        else:
            valid_indices = np.arange(k)
            grads_np_valid = grads_np

        n = grads_np_valid.shape[0]
        cs = _cosine_similarity_matrix(grads_np_valid) - np.eye(n)
        cs_scaled = _standardize_matrix(cs)
        cs_pca = _pca_2d_via_svd(cs_scaled, n_components=int(self.config.fldefender_pca_components))
        centroid = np.median(cs_pca, axis=0, keepdims=True)
        scores = _cosine_similarity_matrix(np.vstack([centroid, cs_pca]))[0, 1:]

        # accumulate reputation
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            self.score_history = np.zeros(self.n_clients, dtype=np.float64)
            self.score_history[valid_indices] = scores

        q1 = float(np.quantile(self.score_history, float(self.config.fldefender_q1)))
        trust = self.score_history - q1
        max_trust = float(trust.max())
        if max_trust > 0:
            trust = trust / max_trust
        trust = np.clip(trust, 0.0, None)
        trust_weights = trust[valid_indices]

        total_weight = float(trust_weights.sum())
        if total_weight > 0:
            w = torch.tensor(trust_weights, dtype=deltas.dtype)
            w = w / (w.sum() + 1e-12)
            delta_agg = (w.unsqueeze(1) * deltas[valid_indices]).sum(dim=0)
        else:
            delta_agg = torch.mean(deltas[valid_indices], dim=0)
        buffer_alpha = torch.zeros(k, dtype=torch.float32)
        if total_weight > 0:
            buffer_alpha[valid_indices] = torch.tensor(trust_weights, dtype=torch.float32)
        else:
            buffer_alpha[valid_indices] = 1.0
        new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
        merged_buffers = _aggregate_nonparam_buffers(global_sd, client_state_dicts, self.param_names, buffer_alpha)
        for kk, vv in merged_buffers.items():
            if kk not in self.param_names:
                new_global_sd[kk] = vv
        self.global_model.load_state_dict(new_global_sd)

        active_ratio = float(np.mean(trust_weights > 0)) if trust_weights.size > 0 else 0.0
        mean_trust = float(np.mean(trust_weights)) if trust_weights.size > 0 else 0.0
        full_trust = np.zeros(k, dtype=np.float64)
        full_trust[valid_indices] = trust_weights
        d = torch.tensor(full_trust, dtype=torch.float32)
        m = (d > 0).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = d.clone()
        if float(alpha.sum().item()) > 0.0:
            alpha = alpha / alpha.sum()
        else:
            alpha = m / (m.sum() + 1e-12)
        self.rounds += 1
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d,
            m=m,
            alpha=alpha,
            phase="fld | FL-Defender",
            show_detection=True,
            monitor_items=[
                ("Defense", "FL-Defender"),
                ("Active ratio", f"{active_ratio:.3f}"),
                ("Mean trust", f"{mean_trust:.3f}"),
                ("Invalid grads", f"{n_invalid}/{k}"),
            ],
            participant_metrics={"trust_score": d.detach().cpu()},
        )


__all__ = ["FLDefenderDefense"]
