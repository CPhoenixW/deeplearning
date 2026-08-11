"""Shared machinery for omniscient, coordinated update-poisoning attacks.

The pipeline exchanges post-local-SGD model states, whereas the original attack
papers formulate their rules over gradients.  A client model delta is the
negative-gradient representation of that quantity, so the callers in this
module deliberately apply the corresponding sign conversion once and keep the
rest of each paper's construction unchanged.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..config import FedConfig
from .base import MaliciousClient


# This keeps the largest temporary ``num_benign x chunk`` tensor modest even
# for the largest convolutional/embedding layers.  It is an implementation
# detail, not an attack hyperparameter.
DELTA_CHUNK_SIZE = 131_072

StateDict = Dict[str, Tensor]


class CoordinatedModelPoisoningAttack(MaliciousClient):
    """Client shell whose actual upload is written by a round-level hook."""

    def local_step(
        self,
        global_state_dict: StateDict,
        reference_state_dict: Optional[StateDict] = None,
    ) -> StateDict:
        if not self.config.skip_redundant_attack_training:
            return super().local_step(global_state_dict, reference_state_dict)
        return reference_state_dict or global_state_dict


def attack_compute_device(config: FedConfig) -> torch.device:
    """Use the aggregation device for attack statistics when it is enabled."""

    if bool(config.cuda_aggregation) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def attack_parameter_names(
    global_state: StateDict,
    parameter_names: Sequence[str] | None,
) -> Tuple[str, ...]:
    """Resolve the trainable state entries attacked by gradient-space methods.

    The normal pipeline supplies ``named_parameters()`` from the global model.
    The floating-state fallback keeps standalone utility calls ergonomic while
    excluding common BatchNorm running-stat buffers, which are not gradients in
    the source attack definitions.
    """

    if parameter_names is None:
        names = tuple(
            name
            for name, value in global_state.items()
            if value.is_floating_point()
            and not name.endswith(("running_mean", "running_var"))
        )
    else:
        names = tuple(str(name) for name in parameter_names)
    if not names:
        raise ValueError("Coordinated model-poisoning attacks require parameters.")
    for name in names:
        if name not in global_state:
            raise KeyError(f"Attack parameter {name!r} is missing from global_state.")
        if not global_state[name].is_floating_point():
            raise TypeError(f"Attack parameter {name!r} must be floating point.")
    return names


def benign_states(config: FedConfig, client_states: Sequence[StateDict]) -> List[StateDict]:
    """Return the known benign uploads used by the omniscient threat model."""

    count = int(config.num_benign)
    if count < 2:
        raise ValueError(
            "Omniscient statistical attacks require at least two benign updates."
        )
    if len(client_states) < count:
        raise ValueError(
            "client_states contains fewer uploads than config.num_benign."
        )
    return list(client_states[:count])


def iter_benign_delta_chunks(
    global_state: StateDict,
    states: Sequence[StateDict],
    parameter_names: Sequence[str],
    *,
    device: torch.device,
    chunk_size: int = DELTA_CHUNK_SIZE,
) -> Iterator[tuple[str, int, int, Tensor]]:
    """Yield ``(parameter, start, end, benign_delta_matrix)`` chunks.

    This is algebraically identical to stacking whole model-update vectors, but
    avoids materializing a ``num_benign x num_parameters`` tensor.  Min-Max and
    Min-Sum can therefore run on ResNet/Transformer models without a multi-GB
    transient allocation.
    """

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive.")
    for name in parameter_names:
        reference = global_state[name].detach().cpu()
        if not reference.is_floating_point():
            raise TypeError(f"Attack parameter {name!r} must be floating point.")
        flat_reference = reference.float().reshape(-1)
        size = int(flat_reference.numel())
        for state_index, state in enumerate(states):
            if name not in state:
                raise KeyError(
                    f"Benign state {state_index} is missing attack parameter {name!r}."
                )
            value = state[name].detach().cpu()
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"State shape mismatch for {name!r}: {tuple(value.shape)} != "
                    f"{tuple(reference.shape)}."
                )
            if not value.is_floating_point():
                raise TypeError(f"Benign state parameter {name!r} must be floating point.")
        for start in range(0, size, int(chunk_size)):
            end = min(size, start + int(chunk_size))
            reference_chunk = flat_reference[start:end].to(device)
            deltas = torch.stack(
                [
                    state[name]
                    .detach()
                    .cpu()
                    .reshape(-1)[start:end]
                    .to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=device.type == "cuda",
                    )
                    - reference_chunk
                    for state in states
                ],
                dim=0,
            )
            if not bool(torch.isfinite(deltas).all().item()):
                raise FloatingPointError(
                    f"Non-finite benign update encountered in attack parameter {name!r}."
                )
            yield name, start, end, deltas


def empty_crafted_delta(global_state: StateDict, parameter_names: Sequence[str]) -> Dict[str, Tensor]:
    """Allocate CPU delta tensors matching the selected model parameters."""

    return {
        name: torch.empty(
            tuple(global_state[name].shape), dtype=torch.float32, device="cpu"
        )
        for name in parameter_names
    }


def rewrite_crafted_uploads(
    global_state: StateDict,
    client_states: List[StateDict],
    malicious_client_ids: Sequence[int],
    parameter_names: Sequence[str],
    crafted_delta: Dict[str, Tensor],
) -> None:
    """Replace selected uploads with one crafted trainable-parameter delta.

    Gradient-space source attacks do not manipulate model buffers.  We therefore
    reset every non-parameter state entry to its global value, rather than
    accidentally retaining locally trained BatchNorm statistics on malicious
    clients.
    """

    client_ids = tuple(int(client_id) for client_id in malicious_client_ids)
    if not client_ids:
        return
    parameter_set = set(parameter_names)
    if set(crafted_delta) != parameter_set:
        raise ValueError("crafted_delta keys must exactly match attack parameter names.")

    template: StateDict = {}
    for name, reference_value in global_state.items():
        reference = reference_value.detach().cpu()
        if name not in parameter_set:
            template[name] = reference.clone()
            continue
        delta = crafted_delta[name].detach().cpu()
        if tuple(delta.shape) != tuple(reference.shape):
            raise ValueError(
                f"Crafted delta shape mismatch for {name!r}: {tuple(delta.shape)} != "
                f"{tuple(reference.shape)}."
            )
        output = reference.float() + delta.float()
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(f"Crafted update is non-finite for {name!r}.")
        template[name] = output.to(dtype=reference.dtype).clone()

    for client_id in client_ids:
        if not 0 <= client_id < len(client_states):
            raise IndexError(f"Malicious client id {client_id} is out of range.")
        client_states[client_id] = {
            name: value.clone() for name, value in template.items()
        }


__all__ = [
    "CoordinatedModelPoisoningAttack",
    "DELTA_CHUNK_SIZE",
    "StateDict",
    "attack_compute_device",
    "attack_parameter_names",
    "benign_states",
    "empty_crafted_delta",
    "iter_benign_delta_chunks",
    "rewrite_crafted_uploads",
]
