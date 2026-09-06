from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class DescriptorLayout:
    output_dim: int
    parameter_count: int
    layer_count: int


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _allocate_layer_dims(total: int, sizes: Sequence[int]) -> List[int]:
    if total < len(sizes):
        # Ultra-low-dimensional descriptors use a shared hash space instead.
        return [0 for _ in sizes]

    allocation = [1 for _ in sizes]
    remaining = total - len(sizes)
    if remaining == 0:
        return allocation

    weights = [math.sqrt(float(size)) for size in sizes]
    weight_sum = sum(weights)
    exact = [remaining * weight / weight_sum for weight in weights]
    floors = [int(value) for value in exact]
    for index, value in enumerate(floors):
        allocation[index] += value

    leftover = remaining - sum(floors)
    order = sorted(
        range(len(sizes)),
        key=lambda index: (exact[index] - floors[index], sizes[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        allocation[index] += 1
    return allocation


class FixedLayerDescriptor:
    """Fixed 4096-D layer-aware CountSketch for federated model updates.

    Each parameter tensor owns a size-weighted section of the descriptor. Bucket
    and sign assignments are deterministic for a model signature and seed, so
    every client and round uses the same projection without a dense matrix.
    """

    def __init__(
        self,
        reference_state_dict: Dict[str, Tensor],
        *,
        parameter_names: Sequence[str],
        output_dim: int = 4096,
        seed: int = 2027,
        projection_device: torch.device | str = "cpu",
    ) -> None:
        if output_dim < 64:
            raise ValueError("output_dim must be at least 64.")
        if not parameter_names:
            raise ValueError("parameter_names must not be empty.")

        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.projection_device = torch.device(projection_device)
        self.parameter_names = tuple(parameter_names)

        shapes: List[Tuple[int, ...]] = []
        sizes: List[int] = []
        for name in self.parameter_names:
            if name not in reference_state_dict:
                raise KeyError(f"Parameter {name!r} is missing from reference_state_dict.")
            value = reference_state_dict[name]
            if not value.is_floating_point():
                raise TypeError(f"Projected parameter {name!r} must be floating point.")
            shapes.append(tuple(value.shape))
            sizes.append(int(value.numel()))

        self._shapes = tuple(shapes)
        self._parameter_count = sum(sizes)
        self.layout = DescriptorLayout(
            output_dim=self.output_dim,
            parameter_count=self._parameter_count,
            layer_count=len(self.parameter_names),
        )

        allocations = _allocate_layer_dims(self.output_dim, sizes)
        shared_hash_space = self.output_dim < len(sizes)
        buckets: List[Tensor] = []
        signs: List[Tensor] = []
        offset = 0
        for name, size, allocated in zip(self.parameter_names, sizes, allocations):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, "layer", name))
            if shared_hash_space:
                buckets.append(
                    torch.randint(
                        0, self.output_dim, (size,), generator=generator, dtype=torch.int32
                    )
                )
            else:
                buckets.append(
                    torch.randint(0, allocated, (size,), generator=generator, dtype=torch.int32)
                    .add_(offset)
                )
                offset += allocated
            signs.append(
                torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
            )

        self._buckets = torch.cat(buckets).to(
            device=self.projection_device, dtype=torch.long
        )
        self._signs = torch.cat(signs).to(
            device=self.projection_device, dtype=torch.float32
        )

    def _validate_state_dict(self, state_dict: Dict[str, Tensor], *, label: str) -> None:
        for name, shape in zip(self.parameter_names, self._shapes):
            if name not in state_dict:
                raise KeyError(f"Parameter {name!r} is missing from {label}.")
            if tuple(state_dict[name].shape) != shape:
                raise ValueError(
                    f"Shape mismatch for {name!r} in {label}: "
                    f"{tuple(state_dict[name].shape)} != {shape}."
                )

    def describe(
        self,
        client_state_dict: Dict[str, Tensor],
        reference_state_dict: Dict[str, Tensor],
    ) -> Tensor:
        self._validate_state_dict(client_state_dict, label="client_state_dict")
        self._validate_state_dict(reference_state_dict, label="reference_state_dict")

        delta = torch.cat(
            [
                (
                    client_state_dict[name].detach().cpu().float()
                    - reference_state_dict[name].detach().cpu().float()
                ).reshape(-1)
                for name in self.parameter_names
            ]
        ).to(self.projection_device, non_blocking=True)
        descriptor = torch.zeros(self.output_dim, device=self.projection_device)
        descriptor.scatter_add_(0, self._buckets, delta * self._signs)
        return descriptor.detach().cpu()

    def describe_many(
        self,
        client_state_dicts: Sequence[Dict[str, Tensor]],
        reference_state_dict: Dict[str, Tensor],
    ) -> Tensor:
        if not client_state_dicts:
            raise ValueError("client_state_dicts must not be empty.")
        return torch.stack(
            [self.describe(state_dict, reference_state_dict) for state_dict in client_state_dicts],
            dim=0,
        )
