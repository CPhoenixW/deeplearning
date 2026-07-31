from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..config import FedConfig
from ..utils import weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats
from .common import _fit_1d_gmm_largest_cluster, _state_l2_distance


class FLGMMDefense(BaseDefense):
    """FLGMM-style GMM/SPC filtering over model-distance statistics."""

    defense_name = "flgmm"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._distance_history: List[List[float]] = []
        self._ucl: Optional[float] = None

    def _ensure_history(self, k: int) -> None:
        if len(self._distance_history) != k:
            self._distance_history = [[] for _ in range(k)]
            self._ucl = None

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        self._ensure_history(k)

        # FLGMM first compares local models against the all-client temporary
        # FedAvg model, then models this 1-D distance distribution.
        temp_global_sd = weighted_fedavg(
            client_state_dicts, torch.full((k,), 1.0 / float(k), dtype=torch.float32)
        )
        raw_dist = np.array(
            [_state_l2_distance(sd, temp_global_sd) for sd in client_state_dicts],
            dtype=np.float64,
        )
        normal_mask, normal_mean, normal_std = _fit_1d_gmm_largest_cluster(
            raw_dist, n_iter=int(self.config.flgmm_em_iters)
        )
        z_dist = (raw_dist - normal_mean) / max(normal_std, 1e-6)
        for i, z in enumerate(z_dist.tolist()):
            self._distance_history[i].append(float(z))

        warmup = max(1, int(self.config.flgmm_warmup_rounds))
        if round_idx <= warmup:
            m_np = normal_mask.astype(np.float32)
            phase = "flgmm | GMM Warm-up"
            if round_idx == warmup:
                all_z = np.array(
                    [z for hist in self._distance_history for z in hist],
                    dtype=np.float64,
                )
                normal_z_mask, _, _ = _fit_1d_gmm_largest_cluster(
                    all_z, n_iter=int(self.config.flgmm_em_iters)
                )
                normal_z = all_z[normal_z_mask]
                if normal_z.size == 0:
                    normal_z = all_z
                self._ucl = float(
                    np.mean(normal_z)
                    + float(self.config.flgmm_control_l) * max(float(np.std(normal_z)), 1e-6)
                )
        else:
            ucl = float(self._ucl) if self._ucl is not None else float(
                np.mean(z_dist) + float(self.config.flgmm_control_l) * max(float(np.std(z_dist)), 1e-6)
            )
            m_np = (z_dist < ucl).astype(np.float32)
            phase = "flgmm | SPC Filtering"

        m = torch.tensor(m_np, dtype=torch.float32)
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = m / (m.sum() + 1e-12)
        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        d = torch.tensor(z_dist, dtype=torch.float32)
        kept = int(m.sum().item())
        ucl_label = "pending" if self._ucl is None else f"{self._ucl:.4f}"
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
                ("Defense", "FLGMM"),
                ("GMM normal mean", f"{normal_mean:.4f}"),
                ("GMM normal std", f"{normal_std:.4f}"),
                ("SPC UCL", ucl_label),
                ("Kept clients", f"{kept}/{k}"),
            ],
            participant_metrics={"standardized_distance": d.detach().cpu()},
        )


__all__ = ["FLGMMDefense"]
