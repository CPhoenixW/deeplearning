from __future__ import annotations

import torch

from .base import BaseDefense, DefenseResult as RoundStats
from ..utils import aggregate_updates_with_info


class TrimmedMeanDefense(BaseDefense):
    defense_name = "tm"

    def _aggregate(self, round_idx, client_state_dicts):
        count = len(client_state_dicts)
        byzantine = self.config.trimmed_mean_num_byzantine
        if byzantine is None:
            byzantine = max(0, self.config.num_clients - self.config.num_benign)
        global_state, accepted, weights = aggregate_updates_with_info(
            client_state_dicts,
            method="trimmed_mean",
            trim_ratio=self.config.trimmed_mean_ratio,
            num_byzantine=byzantine,
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_state)
        return RoundStats(
            center_norm=float("nan"), z_var=0.0, ae_loss=float("nan"),
            svdd_loss=float("nan"), d=torch.zeros(count), m=accepted,
            alpha=weights, phase="aggregation", show_detection=True,
            monitor_items=[
                ("Defense", "Trimmed Mean"),
                ("Byzantine b", str(int(byzantine))),
                ("Kept per coordinate", f"{count - 2 * int(byzantine)}/{count}"),
            ],
            participant_metrics={"aggregation_weight": weights.detach().cpu()},
        )


__all__ = ["TrimmedMeanDefense"]
