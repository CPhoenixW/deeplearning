"""FedDMC LIT targeted backdoor attack.

This implementation follows the authors' released ``LIT_attack`` routine, not
the repository's pre-existing ``lie`` attack.  The two attacks have different
threat models: LIT bootstraps a trigger-trained model from the mean malicious
update and clips its resulting gradient to the malicious-update envelope.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..clients import ModelFactory
from ..config import FedConfig
from .base import MaliciousClient
from .coordinated import StateDict, attack_parameter_names
from .feddmc_backdoor import apply_feddmc_trigger


class LITAttack(MaliciousClient):
    """Initial clean local step followed by the coordinated LIT rewrite."""

    def _train_backdoor_from_mean(self, mean_state: StateDict) -> StateDict:
        """Train the source attack's regularized all-trigger local model."""

        model = self.model_fn().to(self.device)
        model.load_state_dict(mean_state)
        model.train()
        optimizer = self._build_optimizer(model)
        reference_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        batch_size = int(self.config.lit_backdoor_batch_size)
        if batch_size < 1:
            raise ValueError("lit_backdoor_batch_size must be positive.")
        loader = DataLoader(
            self.loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.config.num_workers),
        )
        alpha = float(self.config.lit_regularization)
        target = int(self.config.feddmc_backdoor_target_label)
        for _ in range(int(self.config.local_epochs)):
            for inputs, labels in loader:
                inputs = apply_feddmc_trigger(inputs.to(self.device, non_blocking=True))
                labels = torch.full_like(labels, target, device=self.device)
                if self.config.channels_last:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=bool(self.config.use_amp and self.device.type == "cuda"),
                ):
                    loss = self._criterion(model(inputs), labels)
                    if alpha > 0.0:
                        regularizer = sum(
                            F.mse_loss(parameter, reference_parameters[name])
                            for name, parameter in model.named_parameters()
                        )
                        loss = loss + alpha * regularizer
                if bool(torch.isfinite(loss.detach()).item()):
                    loss.backward()
                    optimizer.step()
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def validate_lit_config(config: FedConfig) -> None:
    attackers = int(config.num_clients) - int(config.num_benign)
    if attackers < 2:
        raise ValueError("LIT requires at least two malicious clients for its standard deviation.")
    if float(config.client_lr) == 0.0:
        raise ValueError("LIT requires a non-zero client_lr.")
    if int(config.lit_backdoor_batch_size) < 1:
        raise ValueError("lit_backdoor_batch_size must be positive.")
    if not math.isfinite(float(config.lit_regularization)) or float(config.lit_regularization) < 0.0:
        raise ValueError("lit_regularization must be a finite non-negative value.")
    if not math.isfinite(float(config.lit_clip_z)) or float(config.lit_clip_z) < 0.0:
        raise ValueError("lit_clip_z must be a finite non-negative value.")


def lit_attack_metadata(config: FedConfig) -> Dict[str, object]:
    return {
        "lit_variant": "FedDMC reference LIT_attack",
        "lit_target_label": int(config.feddmc_backdoor_target_label),
        "lit_backdoor_batch_size": int(config.lit_backdoor_batch_size),
        "lit_regularization": float(config.lit_regularization),
        "lit_clip_z": float(config.lit_clip_z),
        "lit_gradient_statistics": "malicious_preliminary_updates",
    }


def _mean_state(states: Sequence[StateDict], names: Sequence[str]) -> StateDict:
    return {
        name: torch.stack([state[name].detach().cpu().float() for state in states]).mean(dim=0)
        for name in names
    }


def apply_lit_round(
    config: FedConfig,
    _defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
    *,
    attack_clients: Sequence[object] | None,
) -> None:
    """Perform LIT's second local training phase and envelope-constrained upload."""

    validate_lit_config(config)
    if attack_clients is None:
        raise ValueError("LIT requires live attack clients during the coordinated round hook.")
    if len(attack_clients) != len(client_states):
        raise ValueError("LIT attack clients and client states must have the same length.")
    malicious_ids = tuple(range(int(config.num_benign), int(config.num_clients)))
    if not malicious_ids or max(malicious_ids) >= len(client_states):
        raise ValueError("LIT malicious-client IDs are incompatible with client_states.")
    names = attack_parameter_names(global_state, parameter_names)
    preliminary = [client_states[client_id] for client_id in malicious_ids]
    mean_state = _mean_state(preliminary, names)
    backdoor_states: List[StateDict] = []
    for client_id in malicious_ids:
        trainer = getattr(attack_clients[client_id], "_train_backdoor_from_mean", None)
        if not callable(trainer):
            raise TypeError("LIT coordinated hook received a non-LIT malicious client.")
        state = trainer({
            key: (mean_state[key] if key in mean_state else value.detach().cpu().clone())
            for key, value in global_state.items()
        })
        backdoor_states.append(state)

    learning_rate = float(config.client_lr)
    crafted: StateDict = {}
    for name in names:
        global_value = global_state[name].detach().cpu().float()
        preliminary_values = torch.stack(
            [state[name].detach().cpu().float() for state in preliminary]
        )
        gradients = (global_value.unsqueeze(0) - preliminary_values) / learning_rate
        gradient_mean = gradients.mean(dim=0)
        gradient_std = gradients.std(dim=0, unbiased=True)
        malicious_backdoor_mean = torch.stack(
            [state[name].detach().cpu().float() for state in backdoor_states]
        ).mean(dim=0)
        new_parameters = malicious_backdoor_mean + learning_rate * gradient_mean
        new_gradient = (mean_state[name] - new_parameters) / learning_rate
        bounded_gradient = torch.clamp(
            new_gradient,
            min=gradient_mean - float(config.lit_clip_z) * gradient_std,
            max=gradient_mean + float(config.lit_clip_z) * gradient_std,
        )
        crafted[name] = (global_value - learning_rate * bounded_gradient).to(
            dtype=global_state[name].dtype
        )

    for client_id in malicious_ids:
        client_states[client_id] = {
            key: (
                crafted[key].clone()
                if key in crafted
                else value.detach().cpu().clone()
            )
            for key, value in global_state.items()
        }


__all__ = [
    "LITAttack",
    "apply_lit_round",
    "lit_attack_metadata",
    "validate_lit_config",
]
