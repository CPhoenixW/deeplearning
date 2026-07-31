"""Compatibility adapter for the data-free multi-view DMC detector.

The modular pipeline owns the common ``DefenseContext``/``DefenseResult``
contract.  The first implementation of the DMC-style detector lives in the
legacy server module, so this adapter lets both runners use the same detector
without duplicating its numerical implementation.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
from torch import Tensor, nn

from ..config import FedConfig
from ..server import FedDMCServer
from .base import BaseDefense, DefenseResult


class DMCDefense(BaseDefense):
    defense_name = "dmc"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._legacy = FedDMCServer(config, d_bn, device, model_fn)

    def _aggregate(
        self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> DefenseResult:
        stats = self._legacy.aggregate(round_idx, client_state_dicts)
        self.global_model.load_state_dict(self._legacy.global_model.state_dict())
        return DefenseResult(
            center_norm=float(stats.center_norm),
            z_var=float(stats.z_var),
            ae_loss=float(stats.ae_loss),
            svdd_loss=float(stats.svdd_loss),
            d=stats.d.detach().cpu(),
            m=stats.m.detach().cpu(),
            alpha=stats.alpha.detach().cpu(),
            phase=stats.phase,
            show_detection=stats.show_detection,
            monitor_items=list(stats.monitor_items),
            participant_metrics={},
        )


__all__ = ["DMCDefense"]
