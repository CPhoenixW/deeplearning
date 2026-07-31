from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..config import FedConfig
from ..utils import weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats
from .common import _flatten_delta, _mar_forecast, _sample_flat_delta


class FLANDERSDefense(BaseDefense):
    """FLANDERS-style MAR forecast filtering over client parameter time series."""

    defense_name = "flanders"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._history: List[List[np.ndarray]] = []
        self._sample_idx: Optional[Tensor] = None

    def _ensure_state(self, k: int, flat_dim: int) -> None:
        if len(self._history) != k:
            self._history = [[] for _ in range(k)]
        if self._sample_idx is None:
            sample_n = int(self.config.flanders_sampling)
            if sample_n <= 0 or sample_n >= flat_dim:
                self._sample_idx = None
            else:
                gen = torch.Generator()
                gen.manual_seed(int(self.config.seed))
                self._sample_idx = torch.randperm(flat_dim, generator=gen)[:sample_n]

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        first_flat = _flatten_delta(global_sd, client_state_dicts[0])
        self._ensure_state(k, int(first_flat.numel()))

        sampled_updates: List[Tensor] = [first_flat if self._sample_idx is None else first_flat[self._sample_idx]]
        for sd in client_state_dicts[1:]:
            sampled_updates.append(_sample_flat_delta(global_sd, sd, self._sample_idx))

        current = np.stack([u.detach().cpu().numpy().astype(np.float64) for u in sampled_updates], axis=0)
        for i, row in enumerate(current):
            self._history[i].append(row)
            window = int(self.config.flanders_window)
            if window > 0 and len(self._history[i]) > window:
                self._history[i] = self._history[i][-window:]

        min_hist = min(len(h) for h in self._history) if self._history else 0
        if min_hist < 2:
            scores = np.zeros(k, dtype=np.float64)
            m = torch.ones(k, dtype=torch.float32)
            phase = "flanders | Warm-up"
        else:
            hist = np.stack([np.stack(h[-min_hist:], axis=0) for h in self._history], axis=0)
            params_tensor = np.transpose(hist, (0, 2, 1))  # (clients, params, time)
            ground_truth = params_tensor[:, :, -1].copy()
            predicted = _mar_forecast(
                params_tensor[:, :, :-1],
                pred_step=1,
                alpha=float(self.config.flanders_alpha),
                beta=float(self.config.flanders_beta),
                maxiter=int(self.config.flanders_maxiter),
            )[:, :, 0]
            scores = np.sqrt(np.sum((ground_truth - predicted) ** 2, axis=1))
            keep_n_cfg = self.config.flanders_num_clients_to_keep
            keep_n = int(keep_n_cfg) if keep_n_cfg is not None else int(self.config.num_benign)
            keep_n = max(1, min(k, keep_n))
            keep_idx = np.argsort(scores)[:keep_n]
            m_np = np.zeros(k, dtype=np.float32)
            m_np[keep_idx] = 1.0
            m = torch.tensor(m_np, dtype=torch.float32)
            phase = "flanders | MAR Filtering"

        alpha = m / (m.sum() + 1e-12)
        global_sd_new = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd_new)

        d = torch.tensor(scores, dtype=torch.float32)
        kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase=phase,
            show_detection=True,
            monitor_items=[
                ("Defense", "FLANDERS"),
                ("MAR window used", f"{min_hist}"),
                ("Sampled params", f"{int(current.shape[1])}"),
                ("MAR score avg", f"{float(np.mean(scores)):.4f}"),
                ("Kept clients", f"{kept}/{k}"),
            ],
            participant_metrics={"mar_score": d.detach().cpu()},
        )


__all__ = ["FLANDERSDefense"]
