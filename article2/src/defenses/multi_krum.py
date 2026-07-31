from __future__ import annotations

import torch

from .base import BaseDefense, DefenseResult as RoundStats
from ..utils import compute_multi_krum_scores, weighted_fedavg


class MultiKrumDefense(BaseDefense):
    defense_name = "mk"

    def _aggregate(self, round_idx, client_state_dicts):
        count = len(client_state_dicts)
        byzantine = self.config.krum_num_byzantine
        if byzantine is None:
            byzantine = max(0, self.config.num_clients - self.config.num_benign)
        neighbors = count - int(byzantine) - 2
        device_scores = compute_multi_krum_scores(
            client_state_dicts,
            num_byzantine=int(byzantine),
            device=self.aggregation_device,
        )
        selected_count = self.config.multi_krum_num_selected or neighbors
        selected_count = max(1, min(int(selected_count), count))
        selected = torch.topk(device_scores, k=selected_count, largest=False).indices.cpu()
        accepted = torch.zeros(count)
        accepted[selected] = 1.0
        weights = accepted / accepted.sum()
        selected_states = [client_state_dicts[int(i)] for i in selected.tolist()]
        global_state = weighted_fedavg(
            selected_states,
            torch.full((selected_count,), 1.0 / selected_count),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_state)
        return RoundStats(
            center_norm=float("nan"), z_var=0.0, ae_loss=float("nan"),
            svdd_loss=float("nan"), d=device_scores.detach().cpu(), m=accepted,
            alpha=weights, phase="selection", show_detection=True,
            monitor_items=[
                ("Defense", "Multi-Krum"),
                ("Byzantine f", str(int(byzantine))),
                ("Selected clients", f"{selected_count}/{count}"),
            ],
            participant_metrics={"krum_score": device_scores.detach().cpu()},
        )


__all__ = ["MultiKrumDefense"]
