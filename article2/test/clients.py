from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    from .config import FedConfig
except ImportError:
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

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.config.client_lr,
            momentum=self.config.client_momentum,
            weight_decay=self.config.client_weight_decay,
        )

    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Hook for subclasses to modify local training batches."""
        return x, y

    def _train_one_round(self, model: nn.Module) -> None:
        model.train()
        optimizer = self._build_optimizer(model)
        criterion = self._build_criterion()
        for _ in range(self.config.local_epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x, y = self._transform_batch(x, y)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Hook for subclasses to modify uploaded model state dict."""
        return {k: v.detach().cpu().clone() for k, v in local_state_dict.items()}

    def local_step(self, global_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        self._train_one_round(model)
        local_sd = model.state_dict()
        return self._postprocess_upload(global_state_dict, local_sd)


class MaliciousClientBase(BenignClient):
    """Base class for malicious clients with override hooks."""

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        config: FedConfig,
        loader: DataLoader,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(client_id, device, config, loader, model_fn)


class GaussianNoiseClient(MaliciousClientBase):
    """Attack client: adds Gaussian noise to global params and returns."""

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        sigma = self.config.gaussian_sigma
        noisy: Dict[str, Tensor] = {}
        for k, v in global_state_dict.items():
            t = v.detach().cpu()
            if t.is_floating_point():
                noise = torch.randn_like(t) * sigma
                noisy[k] = (t + noise).clone()
            else:
                noisy[k] = t.clone()
        return noisy


class LabelFlippingClient(MaliciousClientBase):
    """Train with symmetric label flip y' = (C-1) - y for C-way classification."""

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        c = int(self.config.num_classes)
        return x, ((c - 1) - y)


class SignFlippingClient(MaliciousClientBase):
    """Train normally, then upload global - scale * (local - global)."""

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        scale = self.config.sign_flip_scale

        flipped: Dict[str, Tensor] = {}
        for k, v_global in global_state_dict.items():
            v_local = local_state_dict[k].detach().cpu()
            g = v_global.detach().cpu()
            if g.is_floating_point():
                flipped[k] = (g - scale * (v_local - g)).clone()
            else:
                flipped[k] = g.clone()
        return flipped


class BackdoorClient(MaliciousClientBase):
    """Backdoor attack client.

    Inject a small square trigger into a fraction of local training samples,
    and relabel them to a target label to implant a backdoor.
    """

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        poison_ratio = float(self.config.backdoor_poison_ratio)
        target = int(self.config.backdoor_target_label)
        s = int(self.config.backdoor_trigger_size)
        val = float(self.config.backdoor_trigger_value)

        if poison_ratio > 0.0 and s > 0:
            mask = torch.rand(y.shape[0], device=self.device) < poison_ratio
            if mask.any():
                x_poison = x.clone()
                y_poison = y.clone()
                x_poison[mask, :, -s:, -s:] = val
                y_poison[mask] = target
                return x_poison, y_poison
        return x, y

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        scale = float(self.config.backdoor_model_replace_scale)
        if scale <= 0:
            scale = 1.0

        attacked: Dict[str, Tensor] = {}
        for k, v_global in global_state_dict.items():
            g = v_global.detach().cpu()
            v_local = local_state_dict[k].detach().cpu()
            if g.is_floating_point():
                attacked[k] = (g + scale * (v_local - g)).clone()
            else:
                attacked[k] = g.clone()
        return attacked


class LieAttackClient(MaliciousClientBase):
    """LIE/ALIE attack client.

    Important: standard ALIE construction needs *all malicious clients'*
    updates in the same round (see `article2/article/experiment.md`).

    Therefore, this client only performs normal local training and uploads
    its model update. The ALIE伪装/重写逻辑 is applied in `new/main.py`
    after collecting all client updates.
    """


ATTACK_REGISTRY: Dict[str, Type[BaseClient]] = {
    "gaussian_noise": GaussianNoiseClient,
    "label_flipping": LabelFlippingClient,
    "sign_flipping": SignFlippingClient,
    "backdoor": BackdoorClient,
}

