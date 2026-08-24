#!/usr/bin/env python3
"""Run the existing FL pipeline with a 4096-dim absolute-parameter AE-SVDD input.

This is an isolated experiment entry point.  It keeps the production
``SVDDDefense`` unchanged and replaces only its feature construction:

    trainable client parameters
        -> deterministic layer-balanced clipping to exactly 4096 coordinates
        -> feature-wise mean/std normalization over finite clients
        -> AE/SVDD scoring
        -> validation-driven Top-K selection and normal SVDD aggregation

The selected client states are still aggregated as model states.  The
clipping and normalization in this file define the detector input; they do
not silently rewrite the model states used by FedAvg.

The 4096 coordinates are taken directly from the absolute trainable parameter
vector.  Quotas are proportional to layer size, so every sufficiently large
layer contributes and the input width is fixed without a ``max-input-dim``
guard or a random descriptor.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Dict, List

import torch
from torch import Tensor

from src.defenses import DEFENSE_REGISTRY
from src.defenses.svdd import SVDDDefense
from src.pipeline import main as pipeline_main
from src.pipeline_core import runner


class AbsoluteParameterSVDDDefense(SVDDDefense):
    """SVDD variant whose AE input is a fixed 4096-dim absolute parameter slice."""

    defense_name = "svdd"
    parameter_dim: int = 4096
    normalization_eps: float = 1e-6

    def __init__(self, config, d_bn, device, model_fn, validation_loader=None, svdd_feature_extractor=None):
        # The parent class creates a fixed descriptor when the normal config
        # says ``fixed_projection``.  This experiment does not need it, while
        # retaining the rest of the parent SVDD implementation unchanged.
        experiment_config = copy.deepcopy(config)
        experiment_config.svdd_feature_mode = "task"
        super().__init__(
            experiment_config,
            d_bn=d_bn,
            device=device,
            model_fn=model_fn,
            validation_loader=validation_loader,
            svdd_feature_extractor=svdd_feature_extractor,
        )
        self._fixed_descriptor = None
        eps = float(type(self).normalization_eps)
        if not torch.isfinite(torch.tensor(eps)) or eps <= 0.0:
            raise ValueError("normalization_eps must be positive and finite.")
        self._parameter_indices = self._build_parameter_indices()

    def _build_parameter_indices(self) -> Tensor:
        """Build deterministic, layer-balanced indices into the flat vector."""

        sizes = [int(self.global_model.get_parameter(name).numel()) for name in self.param_names]
        total = sum(sizes)
        target = int(type(self).parameter_dim)
        if total < target:
            raise ValueError(
                f"The model has only {total:,} trainable parameters; cannot select {target:,}."
            )
        raw_quotas = [target * size / total for size in sizes]
        quotas = [min(size, int(value)) for size, value in zip(sizes, raw_quotas)]
        remainder = target - sum(quotas)
        order = sorted(
            range(len(sizes)),
            key=lambda index: raw_quotas[index] - quotas[index],
            reverse=True,
        )
        for index in order:
            if remainder == 0:
                break
            if quotas[index] < sizes[index]:
                quotas[index] += 1
                remainder -= 1
        if remainder:
            raise RuntimeError("Could not allocate the fixed parameter dimension across layers.")

        indices: list[Tensor] = []
        offset = 0
        for size, quota in zip(sizes, quotas):
            if quota:
                if quota == size:
                    local = torch.arange(size, dtype=torch.long)
                else:
                    # Evenly spaced coordinates avoid over-representing the
                    # first layer elements while keeping the operation deterministic.
                    local = torch.linspace(0, size - 1, quota).round().long()
                indices.append(local + offset)
            offset += size
        result = torch.cat(indices)
        if result.numel() != target or result.unique().numel() != target:
            raise RuntimeError("Fixed parameter clipping produced invalid coordinate indices.")
        return result

    def _absolute_parameter_matrix(
        self, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tensor:
        rows: list[Tensor] = []
        for state_dict in client_state_dicts:
            parts: list[Tensor] = []
            for name in self.param_names:
                value = state_dict[name]
                if not value.is_floating_point():
                    raise TypeError(f"Trainable parameter {name!r} is not floating point.")
                parts.append(value.detach().cpu().float().reshape(-1))
            if not parts:
                raise ValueError("The model has no floating-point trainable parameters.")
            rows.append(torch.cat(parts))
        raw = torch.stack(rows, dim=0)
        return raw.index_select(1, self._parameter_indices)

    def _build_input_matrix(
        self, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tensor:
        """Return exactly 4096 direct absolute trainable parameters per client."""

        raw = self._absolute_parameter_matrix(client_state_dicts)
        finite_rows = torch.isfinite(raw).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All absolute-parameter rows are non-finite.")

        # Invalid rows are kept as zero placeholders.  They remain invalid in
        # ``_scale_input_matrix`` and can never receive aggregation weight.
        return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale_input_matrix(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize clipped absolute parameters using finite-client mean/std."""

        finite_rows = torch.isfinite(raw).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All clipped absolute-parameter rows are non-finite.")

        safe = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = safe[finite_rows]
        mean = valid.mean(dim=0)
        std = valid.std(dim=0, unbiased=False).clamp_min(
            float(type(self).normalization_eps)
        )
        normalized = (safe - mean) / std
        normalized[~finite_rows] = 0.0
        if not bool(torch.isfinite(normalized).all().item()):
            raise FloatingPointError("Absolute-parameter normalization produced non-finite values.")
        return normalized, finite_rows


def _raw_parameter_dimension(context) -> int:
    """Return the fixed AE input width for this experiment."""

    return int(AbsoluteParameterSVDDDefense.parameter_dim)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FL pipeline with 4096-dim, mean/std-normalized absolute parameters as SVDD input."
    )
    parser.add_argument("--config", required=True, help="Pipeline JSON configuration.")
    parser.add_argument(
        "--normalization-eps",
        type=float,
        default=1e-6,
        help="Minimum per-feature standard deviation (default: 1e-6).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.normalization_eps <= 0.0 or not torch.isfinite(torch.tensor(args.normalization_eps)):
        parser.error("--normalization-eps must be positive and finite")

    AbsoluteParameterSVDDDefense.normalization_eps = float(args.normalization_eps)

    # ``run_pipeline`` asks the runner for the AE input dimension before it
    # instantiates the defense.  Patch only this isolated process so the
    # production registry and feature-dimension logic remain unchanged.
    original_feature_dimension = runner._feature_dimension
    runner._feature_dimension = _raw_parameter_dimension
    DEFENSE_REGISTRY["svdd"] = AbsoluteParameterSVDDDefense

    forwarded = [sys.argv[0], "--config", args.config]
    if args.dry_run:
        forwarded.append("--dry-run")
    sys.argv = forwarded
    try:
        return int(pipeline_main() or 0)
    finally:
        runner._feature_dimension = original_feature_dimension


if __name__ == "__main__":
    raise SystemExit(main())
