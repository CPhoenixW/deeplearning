"""Generic federated clients without attack-specific behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import FedConfig


ModelFactory = Callable[[], nn.Module]


class BaseClient(ABC):
    """Abstract client interface for federated learning."""

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        config: FedConfig,
        loader: DataLoader,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.config = config
        self.loader = loader

    @abstractmethod
    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Train from the global model and return one uploaded model state."""


class BenignClient(BaseClient):
    """Standard local SGD client."""

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        config: FedConfig,
        loader: DataLoader,
        model_fn: ModelFactory,
    ) -> None:
        super().__init__(client_id, device, config, loader)
        self.model_fn = model_fn
        self._criterion = nn.CrossEntropyLoss()
        self._amp_enabled = bool(config.use_amp and device.type == "cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.config.client_lr,
            momentum=self.config.client_momentum,
            weight_decay=self.config.client_weight_decay,
        )

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Hook used by data-poisoning attack clients."""

        return x, y

    def _train_one_round(self, model: nn.Module) -> None:
        model.train()
        optimizer = self._build_optimizer(model)
        for _ in range(int(self.config.local_epochs)):
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if self.config.channels_last and inputs.ndim == 4:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                inputs, labels = self._transform_batch(inputs, labels)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self._amp_enabled,
                ):
                    logits = model(inputs)
                    loss = self._criterion(logits, labels)
                if self._amp_enabled:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Hook used by model-poisoning attack clients."""

        return {
            key: value.detach().cpu().clone()
            for key, value in local_state_dict.items()
        }

    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        model = self.model_fn()
        first_parameter = next(model.parameters(), None)
        if first_parameter is not None and first_parameter.device != self.device:
            model = model.to(self.device)
        model.load_state_dict(global_state_dict)
        self._train_one_round(model)
        reference = reference_state_dict or global_state_dict
        return self._postprocess_upload(reference, model.state_dict())


__all__ = ["BaseClient", "BenignClient", "ModelFactory"]
