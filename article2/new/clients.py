from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from config import FedConfig


class BaseClient(ABC):
    """Abstract client interface for federated learning."""

    def __init__(self, client_id: int, device: torch.device, config: FedConfig, loader: DataLoader) -> None:
        self.client_id = client_id
        self.device = device
        self.config = config
        self.loader = loader

    @abstractmethod
    def local_step(self, global_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Receive global model, perform local update, and return local model state_dict."""


class BenignClient(BaseClient):
    """Standard SGD training on local data."""

    def __init__(self, client_id: int, device: torch.device, config: FedConfig, loader: DataLoader, model_fn) -> None:
        super().__init__(client_id, device, config, loader)
        self.model_fn = model_fn

    def _train_one_round(self, model: nn.Module) -> None:
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.client_lr,
            momentum=self.config.client_momentum,
            weight_decay=self.config.client_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.config.local_epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

    def local_step(self, global_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        self._train_one_round(model)
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class GaussianNoiseClient(BaseClient):
    """Attack client: adds Gaussian noise to global params and returns."""

    def local_step(self, global_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sigma = self.config.gaussian_sigma
        noisy: Dict[str, Tensor] = {}
        for k, v in global_state_dict.items():
            t = v.detach().cpu()
            # 只对浮点参数施加高斯噪声，整数等保持不变
            if t.is_floating_point():
                noise = torch.randn_like(t) * sigma
                noisy[k] = (t + noise).clone()
            else:
                noisy[k] = t.clone()
        return noisy


class LabelFlippingClient(BenignClient):
    """Train with flipped labels y = 9 - y."""

    def _train_one_round(self, model: nn.Module) -> None:
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.client_lr,
            momentum=self.config.client_momentum,
            weight_decay=self.config.client_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.config.local_epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = (9 - y).to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()


class SignFlippingClient(BenignClient):
    """Train normally, then upload global - scale * (local - global)."""

    def local_step(self, global_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        from copy import deepcopy

        model = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        self._train_one_round(model)
        local_sd = model.state_dict()
        scale = self.config.sign_flip_scale

        flipped: Dict[str, Tensor] = {}
        for k, v_global in global_state_dict.items():
            v_local = local_sd[k].detach().cpu()
            g = v_global.detach().cpu()
            if g.is_floating_point():
                flipped[k] = (g - scale * (v_local - g)).clone()
            else:
                flipped[k] = g.clone()
        return flipped


ATTACK_REGISTRY: Dict[str, Type[BaseClient]] = {
    "gaussian_noise": GaussianNoiseClient,
    "label_flipping": LabelFlippingClient,
    "sign_flipping": SignFlippingClient,
}

