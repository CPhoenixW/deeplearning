from __future__ import annotations

import torch

from .base import BaseDefense, DefenseResult as RoundStats
from ..utils import aggregate_updates_with_info


class FedAvgDefense(BaseDefense):
    defense_name = "avg"

    def _aggregate(self, round_idx, client_state_dicts):
        count = len(client_state_dicts)
        global_state, accepted, weights = aggregate_updates_with_info(
            client_state_dicts, method="fedavg", device=self.aggregation_device
        )
        self.global_model.load_state_dict(global_state)
        return RoundStats(
            center_norm=float("nan"), z_var=0.0, ae_loss=float("nan"),
            svdd_loss=float("nan"), d=torch.zeros(count), m=accepted,
            alpha=weights, phase="aggregation", show_detection=True,
            monitor_items=[("Defense", "FedAvg"), ("Clients Kept", f"{count}/{count}")],
            participant_metrics={"aggregation_weight": weights.detach().cpu()},
        )


__all__ = ["FedAvgDefense"]
