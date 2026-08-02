"""Data-free multi-view malicious-client detection (FedDMC-style)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from ..config import FedConfig
from ..utils import weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats
from .common import _flatten_param_delta


class DMCDefense(BaseDefense):
    """Fuse magnitude, direction, sign, sparsity and temporal anomaly views.

    The detector remains data-free: it only uses the client model deltas visible
    to the server.  A robust median/MAD threshold creates the hard acceptance
    mask, while the accepted clients receive score-dependent trust weights.
    """

    defense_name = "dmc"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._raw_ema: Optional[Tensor] = None
        self._score_ema: Optional[Tensor] = None

    @staticmethod
    def _robust_z(values: Tensor) -> Tensor:
        values = torch.nan_to_num(
            values.float(), nan=0.0, posinf=1e6, neginf=-1e6
        )
        median = values.median()
        mad = (values - median).abs().median()
        scale = torch.clamp(1.4826 * mad, min=1e-6)
        return ((values - median).abs() / scale).clamp(max=1e6)

    def _views(self, deltas: Tensor) -> Tensor:
        clients, _parameters = deltas.shape
        finite = torch.isfinite(deltas).all(dim=1)
        safe = torch.nan_to_num(
            deltas.float(), nan=0.0, posinf=0.0, neginf=0.0
        )

        center = safe.median(dim=0).values
        center_norm = center.norm().clamp_min(1e-12)
        norms = safe.norm(dim=1)
        norm_view = self._robust_z(norms)

        cosine = (safe @ center) / (
            safe.norm(dim=1).clamp_min(1e-12) * center_norm
        )
        direction_view = self._robust_z(1.0 - cosine.clamp(-1.0, 1.0))

        sign_center = torch.sign(safe.sum(dim=0))
        active = sign_center != 0
        if bool(active.any().item()):
            signs = torch.sign(safe[:, active])
            sign_agreement = (signs * sign_center[active]).mean(dim=1)
            raw_sign = 1.0 - sign_agreement
            sign_view = self._robust_z(raw_sign)
        else:
            raw_sign = torch.zeros(clients, dtype=torch.float32)
            sign_view = torch.zeros(clients, dtype=torch.float32)

        nonzero = (safe.abs() > 1e-12).float().mean(dim=1)
        sparsity_view = self._robust_z(nonzero)

        raw = torch.stack([norms, 1.0 - cosine, raw_sign, nonzero], dim=1)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0)
        if self._raw_ema is None or self._raw_ema.shape != raw.shape:
            temporal_view = torch.zeros(clients, dtype=torch.float32)
            self._raw_ema = raw.detach().clone()
        else:
            temporal_raw = (raw - self._raw_ema).abs().mean(dim=1)
            temporal_view = self._robust_z(temporal_raw)
            decay = min(
                1.0,
                max(0.0, float(getattr(self.config, "dmc_ema_decay", 0.8))),
            )
            self._raw_ema = (
                decay * self._raw_ema + (1.0 - decay) * raw.detach()
            )

        views = torch.stack(
            [
                norm_view,
                direction_view,
                sign_view,
                sparsity_view,
                temporal_view,
            ],
            dim=1,
        )
        views[~finite] = 1e6
        return torch.nan_to_num(views, nan=0.0, posinf=1e6, neginf=0.0)

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        clients = len(client_state_dicts)
        if clients == 0:
            raise ValueError("DMC requires at least one client update")

        global_state = self.state_dict_for_clients()
        deltas = torch.stack(
            [
                _flatten_param_delta(global_state, state, self.param_names)
                for state in client_state_dicts
            ],
            dim=0,
        ).float()
        valid_rows = torch.isfinite(deltas).all(dim=1)
        views = self._views(deltas)

        view_weights = torch.tensor(
            [
                float(getattr(self.config, "dmc_norm_weight", 1.0)),
                float(getattr(self.config, "dmc_direction_weight", 1.0)),
                float(getattr(self.config, "dmc_sign_weight", 1.0)),
                float(getattr(self.config, "dmc_sparsity_weight", 0.5)),
                float(getattr(self.config, "dmc_temporal_weight", 1.0)),
            ],
            dtype=torch.float32,
        ).clamp_min(0.0)
        if float(view_weights.sum().item()) <= 0.0:
            view_weights.fill_(1.0)
        view_weights = view_weights / view_weights.sum()
        score = (views * view_weights.view(1, -1)).sum(dim=1)

        score_decay = min(
            1.0,
            max(
                0.0,
                float(getattr(self.config, "dmc_score_ema_decay", 0.7)),
            ),
        )
        if self._score_ema is None or self._score_ema.numel() != clients:
            self._score_ema = score.detach().clone()
        else:
            self._score_ema = (
                score_decay * self._score_ema
                + (1.0 - score_decay) * score.detach()
            )
        score = self._score_ema.clone()

        warmup = max(0, int(getattr(self.config, "dmc_warmup_rounds", 3)))
        tau = max(0.0, float(getattr(self.config, "dmc_tau", 3.0)))
        median = score.median()
        mad = (score - median).abs().median()
        threshold = median + tau * max(float(1.4826 * mad.item()), 1e-6)
        accepted = torch.ones(clients, dtype=torch.float32)
        if round_idx > warmup and float(mad.item()) > 1e-8:
            accepted = (score <= threshold).float()
        accepted[~valid_rows] = 0.0

        min_keep = max(
            1,
            min(clients, int(getattr(self.config, "dmc_min_keep", 1))),
        )
        if int(accepted.sum().item()) < min_keep:
            accepted.zero_()
            valid_indices = torch.where(valid_rows)[0]
            if valid_indices.numel() > 0:
                keep_indices = valid_indices[
                    torch.argsort(score[valid_indices])[:min_keep]
                ]
            else:
                keep_indices = torch.argsort(score)[:min_keep]
            accepted[keep_indices] = 1.0

        trust = (1.0 / (1.0 + score.clamp_min(0.0))) * accepted
        if float(trust.sum().item()) <= 1e-12:
            trust = accepted.clone()
        participant_weights = trust / trust.sum().clamp_min(1e-12)

        aggregation_states: List[Dict[str, Tensor]] = []
        for row_valid, state in zip(valid_rows.tolist(), client_state_dicts):
            if row_valid:
                aggregation_states.append(state)
                continue
            cleaned: Dict[str, Tensor] = {}
            for name, global_value in global_state.items():
                value = state[name].detach().cpu()
                if value.is_floating_point():
                    cleaned[name] = torch.where(
                        torch.isfinite(value),
                        value,
                        global_value.detach().cpu(),
                    ).clone()
                else:
                    cleaned[name] = global_value.detach().cpu().clone()
            aggregation_states.append(cleaned)

        new_global_state = weighted_fedavg(
            aggregation_states,
            participant_weights.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(new_global_state)

        kept = int(accepted.sum().item())
        phase = (
            "dmc | Warm-up"
            if round_idx <= warmup
            else "dmc | Multi-view Filtering"
        )
        return RoundStats(
            center_norm=float("nan"),
            z_var=float(torch.var(score).item()) if clients > 1 else 0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=score.detach().cpu(),
            m=accepted.detach().cpu(),
            alpha=participant_weights.detach().cpu(),
            phase=phase,
            show_detection=True,
            monitor_items=[
                ("Defense", "FedDMC (multi-view)"),
                ("Score threshold", f"{float(threshold):.4f}"),
                ("Kept clients", f"{kept}/{clients}"),
                ("Views", "norm+direction+sign+sparsity+temporal"),
                ("Score avg", f"{float(score.mean().item()):.4f}"),
            ],
            participant_metrics={
                "anomaly_score": score.detach().cpu(),
                "aggregation_weight": participant_weights.detach().cpu(),
            },
        )


__all__ = ["DMCDefense"]
