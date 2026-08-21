from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..config import FedConfig
from ..utils import state_dict_is_finite


@dataclass(frozen=True)
class DefenseContext:
    """Inputs visible to one defense for one communication round."""

    round_idx: int
    global_state: Dict[str, Tensor]
    client_states: List[Dict[str, Tensor]]


@dataclass
class DefenseResult:
    """Common result contract returned by every defense strategy.

    The short ``d/m/alpha`` names remain as compatibility fields for existing
    experiment code.  New pipeline and reporting code uses the descriptive
    properties below.
    """

    center_norm: float
    z_var: float
    ae_loss: float
    svdd_loss: float
    d: Tensor
    m: Tensor
    alpha: Tensor
    phase: str
    show_detection: bool
    monitor_items: List[Tuple[str, str]]
    recon_loss: float = float("nan")
    total_loss: float = float("nan")
    global_state: Dict[str, Tensor] = field(default_factory=dict)
    server_metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    participant_metrics: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def participant_scores(self) -> Tensor:
        return self.d

    @property
    def participant_weights(self) -> Tensor:
        return self.alpha

    @property
    def accepted_mask(self) -> Tensor:
        return self.m


class DefenseStrategy(Protocol):
    defense_name: str

    def aggregate(self, context: DefenseContext) -> DefenseResult:
        ...


class BaseDefense:
    """Shared model ownership and result validation for server defenses."""

    defense_name = "base"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
        validation_loader: DataLoader | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.d_bn = d_bn
        self.validation_loader = validation_loader
        self.global_model = model_fn().to(self.device)
        self.param_names: List[str] = [
            name
            for name, parameter in self.global_model.named_parameters()
            if parameter.requires_grad
        ]
        self._last_finite_global_state = {
            key: value.detach().cpu().clone()
            for key, value in self.global_model.state_dict().items()
        }
        self._global_state_rolled_back = False

    def state_dict_for_clients(self) -> Dict[str, Tensor]:
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.global_model.state_dict().items()
        }
        if state_dict_is_finite(state):
            self._last_finite_global_state = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }
            self._global_state_rolled_back = False
            return state

        # Keep a bad aggregation from becoming the starting point for every
        # client in the next communication round.  The last finite model is a
        # deterministic no-progress fallback and is preferable to letting all
        # local updates become NaN/Inf.
        self.global_model.load_state_dict(self._last_finite_global_state)
        self._global_state_rolled_back = True
        return {
            key: value.detach().cpu().clone()
            for key, value in self._last_finite_global_state.items()
        }

    @property
    def aggregation_device(self) -> torch.device:
        if self.config.cuda_aggregation and self.device.type == "cuda":
            return self.device
        return torch.device("cpu")

    def aggregate(self, context: DefenseContext) -> DefenseResult:
        """Execute one communication round through the canonical context API."""

        result = self._aggregate(context.round_idx, context.client_states)
        return self._finalize_result(result)

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> DefenseResult:
        raise NotImplementedError

    def _finalize_result(self, result: DefenseResult) -> DefenseResult:
        count = len(result.d)
        if result.m.reshape(-1).numel() != count:
            raise ValueError("accepted_mask length does not match participant_scores")
        if result.alpha.reshape(-1).numel() != count:
            raise ValueError("participant_weights length does not match participant_scores")
        for name, values in result.participant_metrics.items():
            if values.reshape(-1).numel() != count:
                raise ValueError(
                    f"participant metric {name!r} length does not match participant_scores"
                )
        result.global_state = self.state_dict_for_clients()
        result.server_metrics.update(
            {
                "center_norm": result.center_norm,
                "z_variance": result.z_var,
                "ae_loss": result.ae_loss,
                "svdd_loss": result.svdd_loss,
                "recon_loss": result.recon_loss,
                "total_loss": result.total_loss,
                "kept": int((result.m >= 0.5).sum().item()),
                "num_clients": count,
                "global_state_rolled_back": float(self._global_state_rolled_back),
            }
        )
        return result


__all__ = [
    "BaseDefense",
    "DefenseContext",
    "DefenseResult",
    "DefenseStrategy",
]
