from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
from torch import Tensor, nn

from ..config import FedConfig
from .base import BaseDefense, DefenseResult as RoundStats
from .common import (
    _aggregate_nonparam_buffers,
    _apply_flat_param_delta_to_global,
    _flatten_param_delta,
    _lasa_layer_dims_from_model,
    _topk_sparsification,
)


class LASADefense(BaseDefense):
    """Port of LASA (WACV 2025) from experiment/FL-Byzantine-Library/aggregators/lasa.py.

    Operates on flattened client deltas: Δ_k = w_k - w_g.
    """

    defense_name = "lasa"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self.layer_dims = _lasa_layer_dims_from_model(self.global_model, self.param_names)

    def _gradient_sanitization_and_clipping(self, updates: Tensor) -> Tensor:
        # updates: (K, D)
        # Mirror library behavior: median-norm clipping.
        grad_norm = torch.norm(updates, dim=1, keepdim=True)  # (K,1)
        norm_clip = torch.median(grad_norm, dim=0).values.item()
        grad_norm_clipped = torch.clamp(grad_norm, 0.0, float(norm_clip))
        grads_clip = (updates / (grad_norm + 1e-12)) * grad_norm_clipped
        return grads_clip

    def _byzantine_detection_layer(self, sparse_updates: List[Tensor], start_dim: int, end_dim: int) -> List[int]:
        all_set = set(range(len(sparse_updates)))
        layer_flat_params: List[Tensor] = [u[start_dim:end_dim] for u in sparse_updates]
        if len(layer_flat_params) == 0:
            return list(all_set)
        grads = torch.stack(layer_flat_params, dim=0)

        # Norm check (MZ-score via median/std)
        grad_l2norm = torch.norm(grads.float(), dim=1).detach().cpu().numpy()
        norm_med = float(np.median(grad_l2norm))
        norm_std = float(np.std(grad_l2norm))
        norm_scores = []
        for v in grad_l2norm:
            score = float(abs((float(v) - norm_med) / norm_std)) if norm_std > 0 else 0.0
            norm_scores.append(score)
        benign_idx1 = all_set.intersection(set(np.argwhere(np.array(norm_scores) < float(self.config.lasa_lambda_n)).flatten().astype(int).tolist()))

        # Sign check
        layer_signs: List[float] = []
        for layer_param in layer_flat_params:
            sign_sum = torch.sum(torch.sign(layer_param))
            abs_sign_sum = torch.sum(torch.abs(torch.sign(layer_param)))
            if abs_sign_sum > 0:
                sign_ratio = 0.5 * (1 + sign_sum / abs_sign_sum * (1 - float(self.config.lasa_sparsity_ratio)))
                layer_signs.append(float(sign_ratio.item()))
            else:
                layer_signs.append(0.5)

        benign_idx2 = all_set.copy()
        sign_scores: List[float] = []
        if len(layer_signs) > 0:
            median_sign = float(np.median(layer_signs))
            std_sign = float(np.std(layer_signs))
            for sign in layer_signs:
                score = float(abs((float(sign) - median_sign) / std_sign)) if std_sign > 0 else 0.0
                sign_scores.append(score)
            benign_idx2 = benign_idx2.intersection(set(np.argwhere(np.array(sign_scores) < float(self.config.lasa_lambda_s)).flatten().astype(int).tolist()))

        benign_indices = list(benign_idx1.intersection(benign_idx2))
        if len(benign_indices) == 0:
            benign_indices = list(all_set)
        return benign_indices

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        )  # (K,D)

        # Step 1: gradient clipping (median norm)
        clipped = self._gradient_sanitization_and_clipping(deltas)

        # Step 2: pre-aggregation sparsification on each client's clipped update
        sparse_updates = [_topk_sparsification(u, float(self.config.lasa_sparsity_ratio)) for u in clipped]

        # Step 3: layer-wise detection + mean over benign indices, concatenate
        aggregated_layers: List[Tensor] = []
        benign_fracs: List[float] = []
        benign_counts = torch.zeros(k, dtype=torch.float32)
        for i in range(int(len(self.layer_dims) - 1)):
            start_dim = int(self.layer_dims[i])
            end_dim = int(self.layer_dims[i + 1])
            benign_indices = self._byzantine_detection_layer(sparse_updates, start_dim, end_dim)
            benign_fracs.append(len(benign_indices) / max(1, k))
            if benign_indices:
                benign_counts[torch.tensor(benign_indices, dtype=torch.long)] += 1.0
            benign_layer_params = [clipped[idx][start_dim:end_dim] for idx in benign_indices if idx < k]
            if len(benign_layer_params) > 0:
                aggregated_layer = torch.mean(torch.stack(benign_layer_params, dim=0), dim=0)
            else:
                aggregated_layer = clipped[0][start_dim:end_dim]
            aggregated_layers.append(aggregated_layer)
        delta_agg = torch.cat(aggregated_layers, dim=0)
        # Use per-client benign-layer ratio as buffer-aggregation weights.
        buffer_alpha = benign_counts / (benign_counts.sum() + 1e-12)
        if float(buffer_alpha.sum().item()) <= 0.0:
            buffer_alpha = torch.full((k,), 1.0 / max(1, k), dtype=torch.float32)
        new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
        merged_buffers = _aggregate_nonparam_buffers(global_sd, client_state_dicts, self.param_names, buffer_alpha)
        for kk, vv in merged_buffers.items():
            if kk not in self.param_names:
                new_global_sd[kk] = vv
        self.global_model.load_state_dict(new_global_sd)

        avg_benign_frac = float(np.mean(benign_fracs)) if benign_fracs else 1.0
        num_layers = max(1, int(len(self.layer_dims) - 1))
        d = benign_counts / float(num_layers)  # per-client benign-layer ratio
        m = (d >= 0.5).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        # Report per-client weight using the same normalized benign-layer count
        # used by buffer aggregation; this is more informative than hard-mask uniform.
        alpha = buffer_alpha
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d,
            m=m,
            alpha=alpha,
            phase="lasa | LASA",
            show_detection=True,
            monitor_items=[
                ("Defense", "LASA"),
                ("Avg benign frac/layer", f"{avg_benign_frac:.3f}"),
                ("Kept clients", f"{int(m.sum().item())}/{k}"),
            ],
            participant_metrics={"benign_layer_ratio": d.detach().cpu()},
        )


__all__ = ["LASADefense"]
