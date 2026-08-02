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
    global_dim: int
    layer_dim: int
    statistics_dim: int
    parameter_count: int
    layer_count: int


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _allocate_layer_dims(total: int, sizes: Sequence[int]) -> List[int]:
    if total == 0:
        return [0 for _ in sizes]
    if total < len(sizes):
        # The caller switches to a shared, layer-seeded hash space when an
        # ultra-low-dimensional descriptor cannot reserve a disjoint bucket
        # for every parameter tensor.
        return [0 for _ in sizes]

    allocation = [1 for _ in sizes]
    remaining = total - len(sizes)
    if remaining == 0:
        return allocation

    weights = [math.sqrt(float(size)) for size in sizes]
    weight_sum = sum(weights)
    exact = [remaining * weight / weight_sum for weight in weights]
    floors = [int(value) for value in exact]
    for idx, value in enumerate(floors):
        allocation[idx] += value

    leftover = remaining - sum(floors)
    order = sorted(
        range(len(sizes)),
        key=lambda idx: (exact[idx] - floors[idx], sizes[idx], -idx),
        reverse=True,
    )
    for idx in order[:leftover]:
        allocation[idx] += 1
    return allocation


def _allocate_view_dims(
    total: int,
    ratios: Sequence[float],
) -> Tuple[int, ...]:
    """Allocate an exact descriptor budget with largest remainders."""

    if total <= 0:
        raise ValueError("Descriptor output dimension must be positive.")
    if len(ratios) != 3:
        raise ValueError("Exactly three descriptor view ratios are required.")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Descriptor view ratios must be finite and non-negative.")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Descriptor view ratios must sum to 1.0.")

    exact = [total * value for value in values]
    allocated = [int(math.floor(value)) for value in exact]
    remaining = total - sum(allocated)
    order = sorted(
        range(len(values)),
        key=lambda idx: (exact[idx] - allocated[idx], values[idx], -idx),
        reverse=True,
    )
    for idx in order[:remaining]:
        allocated[idx] += 1
    for ratio, dimension in zip(values, allocated):
        if ratio > 0.0 and dimension == 0:
            raise ValueError(
                "A positive descriptor view ratio received zero coordinates; "
                "increase output_dim or set the ratio to zero."
            )
    return tuple(allocated)


class FixedHierarchicalMultiViewDescriptor:
    """Fixed descriptor Phi(delta W) for high-dimensional federated updates.

    The two parameter views are implicit CountSketch matrices. Their bucket and
    sign assignments are fixed by the model signature and seed, so every round
    and client uses exactly the same mapping without materializing a dense
    ``output_dim x parameter_count`` matrix.
    """

    _STATISTICS_PER_LAYER = 6

    def __init__(
        self,
        reference_state_dict: Dict[str, Tensor],
        *,
        parameter_names: Sequence[str],
        output_dim: int = 4096,
        seed: int = 2027,
        projection_device: torch.device | str = "cpu",
        global_ratio: float = 0.5,
        layer_ratio: float = 0.375,
        statistics_ratio: float = 0.125,
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
        dtypes: List[torch.dtype] = []
        for name in self.parameter_names:
            if name not in reference_state_dict:
                raise KeyError(f"Parameter {name!r} is missing from reference_state_dict.")
            value = reference_state_dict[name]
            if not value.is_floating_point():
                raise TypeError(f"Projected parameter {name!r} must be floating point.")
            shapes.append(tuple(value.shape))
            sizes.append(int(value.numel()))
            dtypes.append(value.dtype)

        self._shapes = tuple(shapes)
        self._sizes = tuple(sizes)
        self._dtypes = tuple(dtypes)
        self._parameter_count = sum(sizes)

        global_dim, layer_dim, statistics_dim = _allocate_view_dims(
            self.output_dim,
            (global_ratio, layer_ratio, statistics_ratio),
        )
        layer_allocations = _allocate_layer_dims(layer_dim, sizes)
        self.layout = DescriptorLayout(
            output_dim=self.output_dim,
            global_dim=global_dim,
            layer_dim=layer_dim,
            statistics_dim=statistics_dim,
            parameter_count=self._parameter_count,
            layer_count=len(self.parameter_names),
        )

        global_buckets: List[Tensor] = []
        global_signs: List[Tensor] = []
        layer_buckets: List[Tensor] = []
        layer_signs: List[Tensor] = []
        layer_offset = 0
        shared_layer_space = 0 < layer_dim < len(sizes)
        for name, size, allocated in zip(self.parameter_names, sizes, layer_allocations):
            generator = torch.Generator(device="cpu")
            if global_dim > 0:
                generator.manual_seed(_stable_seed(self.seed, "global", name))
                global_buckets.append(
                    torch.randint(0, global_dim, (size,), generator=generator, dtype=torch.int32)
                )
                global_signs.append(
                    torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
                )

            if layer_dim > 0:
                generator.manual_seed(_stable_seed(self.seed, "layer", name))
                if shared_layer_space:
                    layer_buckets.append(
                        torch.randint(0, layer_dim, (size,), generator=generator, dtype=torch.int32)
                    )
                else:
                    layer_buckets.append(
                        torch.randint(0, allocated, (size,), generator=generator, dtype=torch.int32)
                        .add_(layer_offset)
                    )
                layer_signs.append(
                    torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
                )
            if not shared_layer_space:
                layer_offset += allocated

        self._projection_tensors = (
            (
                torch.cat(global_buckets).to(device=self.projection_device, dtype=torch.long)
                if global_buckets else torch.empty(0, device=self.projection_device, dtype=torch.long)
            ),
            (
                torch.cat(global_signs).to(device=self.projection_device, dtype=torch.float32)
                if global_signs else torch.empty(0, device=self.projection_device, dtype=torch.float32)
            ),
            (
                torch.cat(layer_buckets).to(device=self.projection_device, dtype=torch.long)
                if layer_buckets else torch.empty(0, device=self.projection_device, dtype=torch.long)
            ),
            (
                torch.cat(layer_signs).to(device=self.projection_device, dtype=torch.float32)
                if layer_signs else torch.empty(0, device=self.projection_device, dtype=torch.float32)
            ),
        )

        stats_input_dim = len(self.parameter_names) * self._STATISTICS_PER_LAYER
        if statistics_dim > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, "statistics"))
            statistics_matrix = torch.randn(
                statistics_dim,
                stats_input_dim,
                generator=generator,
                dtype=torch.float32,
            )
            statistics_matrix.mul_(1.0 / math.sqrt(float(statistics_dim)))
            self._statistics_matrix = statistics_matrix.to(device=self.projection_device)
        else:
            self._statistics_matrix = torch.empty(
                (0, stats_input_dim),
                device=self.projection_device,
                dtype=torch.float32,
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

    @staticmethod
    def _layer_statistics(delta: Tensor) -> Tensor:
        if delta.numel() == 0:
            return torch.zeros(6, dtype=torch.float32)
        abs_delta = delta.abs()
        return torch.stack(
            [
                delta.mean(),
                delta.square().mean().sqrt(),
                abs_delta.mean(),
                abs_delta.max(),
                delta.sign().mean(),
                (delta != 0).float().mean(),
            ]
        ).float()

    def describe(
        self,
        client_state_dict: Dict[str, Tensor],
        reference_state_dict: Dict[str, Tensor],
    ) -> Tensor:
        self._validate_state_dict(client_state_dict, label="client_state_dict")
        self._validate_state_dict(reference_state_dict, label="reference_state_dict")

        flat_parts: List[Tensor] = []
        statistics: List[Tensor] = []
        for name in self.parameter_names:
            client = client_state_dict[name].detach().cpu().float()
            reference = reference_state_dict[name].detach().cpu().float()
            delta = (client - reference).reshape(-1)
            if self.layout.global_dim > 0 or self.layout.layer_dim > 0:
                flat_parts.append(delta)
            if self.layout.statistics_dim > 0:
                statistics.append(self._layer_statistics(delta))

        flat_delta = (
            torch.cat(flat_parts).to(self.projection_device, non_blocking=True)
            if flat_parts else torch.empty(0, device=self.projection_device)
        )
        stats_vector = (
            torch.cat(statistics).to(self.projection_device, non_blocking=True)
            if statistics else torch.empty(0, device=self.projection_device)
        )
        global_bucket, global_sign, layer_bucket, layer_sign = self._projection_tensors

        global_view = torch.zeros(self.layout.global_dim, device=self.projection_device)
        if self.layout.global_dim > 0:
            global_view.scatter_add_(0, global_bucket, flat_delta * global_sign)

        layer_view = torch.zeros(self.layout.layer_dim, device=self.projection_device)
        if self.layout.layer_dim > 0:
            layer_view.scatter_add_(0, layer_bucket, flat_delta * layer_sign)
        statistics_view = (
            self._statistics_matrix @ stats_vector
            if self.layout.statistics_dim > 0
            else torch.empty(0, device=self.projection_device)
        )

        descriptor = torch.cat([global_view, layer_view, statistics_view])
        if descriptor.numel() != self.output_dim:
            raise RuntimeError(
                f"Descriptor dimension mismatch: {descriptor.numel()} != {self.output_dim}."
            )
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
