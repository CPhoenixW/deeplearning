"""FedDMC LIT targeted backdoor attack.

This module ports the LIT implementation released with FedDMC
(Mu et al., IEEE TDSC 2024).  It is intentionally kept separate from the
standard LIE/ALIE attack in :mod:`src.attacks.lie`: FedDMC's LIT first obtains
ordinary local updates from the malicious clients, trains a coordinated
backdoor model from their mean model, and finally clips the resulting gradient
coordinate-wise into ``mean ± z * std`` before assigning the same crafted
upload to every malicious client.

Reference implementation:
https://github.com/MuXutong/FedDMC/blob/main/attack.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from ..clients import BaseClient, BenignClient
from ..config import FedConfig
from .backdoor import BackdoorAttack, evaluate_backdoor_attack
from .coordinated import (
    DELTA_CHUNK_SIZE,
    StateDict,
    attack_parameter_names,
    empty_crafted_delta,
    rewrite_crafted_uploads,
)


class LitAttack(BackdoorAttack):
    """FedDMC LIT client shell.

    The first local step is deliberately benign.  FedDMC computes the LIT
    reference mean/std from the malicious clients' ordinary local updates and
    only then performs a second, coordinated backdoor-training pass from the
    mean malicious model.  ``apply_lit_round`` performs that second pass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lit_backdoor_training = False

    def _transform_batch(self, x: Tensor, y: Tensor):
        if not self._lit_backdoor_training:
            return x, y

        poison_ratio = float(self.config.lit_backdoor_poison_ratio)
        target = int(self.config.backdoor_target_label)
        trigger_size = int(self.config.backdoor_trigger_size)
        trigger_value = float(self.config.backdoor_trigger_value)

        if poison_ratio <= 0.0 or trigger_size <= 0:
            return x, y
        mask = torch.rand(y.shape[0], device=self.device) < poison_ratio
        if not bool(mask.any().item()):
            return x, y

        poisoned_labels = y.clone()
        if x.ndim == 4:
            poisoned_inputs = x.clone()
            poisoned_inputs[mask, :, -trigger_size:, -trigger_size:] = trigger_value
        else:
            # Keep the same behavior as the repository's generic backdoor
            # attack for non-image tasks: targeted-label poisoning only.
            poisoned_inputs = x
        poisoned_labels[mask] = target
        return poisoned_inputs, poisoned_labels

    def _postprocess_upload(
        self,
        global_state_dict: Dict[str, Tensor],
        local_state_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # FedDMC LIT does not use the generic model-replacement scale.  Its
        # strength is controlled by the final coordinate-wise statistical clip.
        del global_state_dict
        return {
            key: value.detach().cpu().clone()
            for key, value in local_state_dict.items()
        }

    def local_step(
        self,
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Dict[str, Tensor] | None = None,
    ) -> Dict[str, Tensor]:
        """Produce the ordinary pre-attack local update used for LIT statistics."""

        self._lit_backdoor_training = False
        return BenignClient.local_step(
            self,
            global_state_dict,
            reference_state_dict=reference_state_dict,
        )

    def backdoor_step(self, start_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Train one backdoor model from the shared FedDMC mean model."""

        self._lit_backdoor_training = True
        try:
            return BenignClient.local_step(
                self,
                start_state_dict,
                reference_state_dict=start_state_dict,
            )
        finally:
            self._lit_backdoor_training = False


def validate_lit_config(config: FedConfig) -> None:
    malicious = int(config.num_clients) - int(config.num_benign)
    if malicious < 2:
        raise ValueError("FedDMC LIT requires at least two malicious clients.")
    z = float(config.lit_z)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("lit_z must be a finite non-negative value.")
    poison_ratio = float(config.lit_backdoor_poison_ratio)
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("lit_backdoor_poison_ratio must be in [0, 1].")
    if float(config.client_lr) <= 0.0:
        raise ValueError("FedDMC LIT requires client_lr > 0.")


def lit_attack_metadata(config: FedConfig) -> Dict[str, object]:
    return {
        "lit_variant": "FedDMC LIT (Mu et al., IEEE TDSC 2024)",
        "lit_coordinate_space": "gradient",
        "lit_z": float(config.lit_z),
        "lit_backdoor_poison_ratio": float(config.lit_backdoor_poison_ratio),
        "lit_target_label": int(config.backdoor_target_label),
        "lit_trigger_size": int(config.backdoor_trigger_size),
    }


def _parameter_stats_from_malicious_updates(
    global_state: StateDict,
    client_states: Sequence[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str],
    *,
    learning_rate: float,
) -> tuple[StateDict, Dict[str, Tensor], Dict[str, Tensor]]:
    """Return FedDMC ``params_mean``, gradient mean and sample std.

    The released FedDMC code computes ``(global - local) / lr`` for each
    malicious client's ordinary local model, then uses ``torch.std`` (sample
    standard deviation by default).  The implementation below is layer/chunk
    equivalent but avoids a full ``M x num_parameters`` temporary tensor.
    """

    params_mean: StateDict = {
        name: value.detach().cpu().clone() for name, value in global_state.items()
    }
    grads_mean: Dict[str, Tensor] = {}
    grads_std: Dict[str, Tensor] = {}

    for name in parameter_names:
        reference = global_state[name].detach().cpu().float()
        mean_tensor = torch.empty_like(reference)
        std_tensor = torch.empty_like(reference)
        mean_param = torch.empty_like(reference)
        flat_ref = reference.reshape(-1)
        flat_mean = mean_tensor.reshape(-1)
        flat_std = std_tensor.reshape(-1)
        flat_param = mean_param.reshape(-1)

        for start in range(0, int(flat_ref.numel()), DELTA_CHUNK_SIZE):
            end = min(int(flat_ref.numel()), start + DELTA_CHUNK_SIZE)
            local_chunk = torch.stack(
                [
                    client_states[int(client_id)][name]
                    .detach()
                    .cpu()
                    .float()
                    .reshape(-1)[start:end]
                    for client_id in malicious_client_ids
                ],
                dim=0,
            )
            grad_chunk = (flat_ref[start:end].unsqueeze(0) - local_chunk) / learning_rate
            mean_chunk = grad_chunk.mean(dim=0)
            std_chunk = grad_chunk.std(dim=0, unbiased=True)
            flat_mean[start:end] = mean_chunk
            flat_std[start:end] = std_chunk
            flat_param[start:end] = flat_ref[start:end] - learning_rate * mean_chunk

        if not bool(torch.isfinite(mean_tensor).all().item()):
            raise FloatingPointError(f"Non-finite FedDMC LIT mean for {name!r}.")
        if not bool(torch.isfinite(std_tensor).all().item()):
            raise FloatingPointError(f"Non-finite FedDMC LIT std for {name!r}.")
        grads_mean[name] = mean_tensor
        grads_std[name] = std_tensor
        params_mean[name] = mean_param.to(dtype=global_state[name].dtype)

    return params_mean, grads_mean, grads_std


def rewrite_lit_uploads(
    config: FedConfig,
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    clients: Sequence[BaseClient],
    parameter_names: Sequence[str] | None = None,
) -> None:
    """Apply the released FedDMC LIT construction to selected malicious clients."""

    validate_lit_config(config)
    client_ids = tuple(int(client_id) for client_id in malicious_client_ids)
    if not client_ids:
        return
    if len(clients) != len(client_states):
        raise ValueError("FedDMC LIT requires one client object per client state.")

    names = attack_parameter_names(global_state, parameter_names)
    lr = float(config.client_lr)
    params_mean, grads_mean, grads_std = _parameter_stats_from_malicious_updates(
        global_state,
        client_states,
        client_ids,
        names,
        learning_rate=lr,
    )

    # FedDMC's train_malicious_network() starts every malicious client from the
    # same params_mean, performs backdoor local training, then averages those
    # backdoored models.
    backdoor_states: List[StateDict] = []
    for client_id in client_ids:
        client = clients[client_id]
        if not isinstance(client, LitAttack):
            raise TypeError(
                "FedDMC LIT round hook requires LitAttack client objects; "
                f"client {client_id} is {type(client).__name__}."
            )
        backdoor_states.append(client.backdoor_step(params_mean))

    z = float(config.lit_z)
    crafted_delta = empty_crafted_delta(global_state, names)

    for name in names:
        reference = global_state[name].detach().cpu().float().reshape(-1)
        params_mean_flat = params_mean[name].detach().cpu().float().reshape(-1)
        grads_mean_flat = grads_mean[name].reshape(-1)
        grads_std_flat = grads_std[name].reshape(-1)
        crafted_flat = crafted_delta[name].reshape(-1)

        for start in range(0, int(reference.numel()), DELTA_CHUNK_SIZE):
            end = min(int(reference.numel()), start + DELTA_CHUNK_SIZE)
            mal_net_chunk = torch.stack(
                [
                    state[name].detach().cpu().float().reshape(-1)[start:end]
                    for state in backdoor_states
                ],
                dim=0,
            ).mean(dim=0)

            # Literal algebra from FedDMC/attack.py:
            #   new_params = mal_net_params + lr * grads_mean
            #   new_grads  = (params_mean - new_params) / lr
            #   new_grads  = clip(new_grads, mean-z*std, mean+z*std)
            #   mal_params = original_params - lr * new_grads
            new_params = mal_net_chunk + lr * grads_mean_flat[start:end]
            new_grads = (params_mean_flat[start:end] - new_params) / lr
            lower = grads_mean_flat[start:end] - z * grads_std_flat[start:end]
            upper = grads_mean_flat[start:end] + z * grads_std_flat[start:end]
            clipped_grads = torch.maximum(torch.minimum(new_grads, upper), lower)
            mal_params = reference[start:end] - lr * clipped_grads
            crafted_flat[start:end] = mal_params - reference[start:end]

    rewrite_crafted_uploads(
        global_state,
        client_states,
        client_ids,
        names,
        crafted_delta,
    )


def apply_lit_round(
    config: FedConfig,
    defense_name: str,
    global_state: StateDict,
    client_states: List[StateDict],
    parameter_names: Sequence[str] | None = None,
    *,
    clients: Sequence[BaseClient] | None = None,
) -> None:
    del defense_name
    if clients is None:
        raise ValueError("FedDMC LIT requires the round's client objects.")
    rewrite_lit_uploads(
        config,
        global_state,
        client_states,
        range(config.num_benign, config.num_clients),
        clients,
        parameter_names,
    )


# Re-export the generic trigger-based ASR evaluator under a LIT-specific name
# for clarity at registration sites.
evaluate_lit_attack = evaluate_backdoor_attack


__all__ = [
    "LitAttack",
    "apply_lit_round",
    "evaluate_lit_attack",
    "lit_attack_metadata",
    "rewrite_lit_uploads",
    "validate_lit_config",
]
