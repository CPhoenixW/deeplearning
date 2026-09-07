"""FedDMC-style random label-flipping data-poisoning attack."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..config import FedConfig
from .base import MaliciousClient


class LabelFlippingAttack(MaliciousClient):
    """Replace every label with a uniformly sampled *different* class.

    FedDMC defines LF as randomly changing each manipulated sample's label to
    a member of the label set other than its original class.  Drawing from
    ``[0, C - 2]`` and shifting values at or above ``y`` produces that exact
    uniform distribution without rejection sampling.  It also remains
    reproducible under the pipeline's global Torch seed.
    """

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        classes = int(self.config.num_classes)
        if classes < 2:
            raise ValueError("Label flipping requires at least two classes.")
        if bool(((y < 0) | (y >= classes)).any().item()):
            raise ValueError("Labels must lie in [0, num_classes) for label flipping.")
        sampled = torch.randint(
            classes - 1,
            y.shape,
            device=y.device,
            dtype=y.dtype,
        )
        return x, sampled + (sampled >= y).to(dtype=y.dtype)


def label_flipping_metadata(_config: FedConfig) -> Dict[str, str]:
    """Record the stochastic LF definition with every experiment result."""

    return {
        "label_flipping_variant": "uniform_random_other_class",
        "label_flipping_reference": "FedDMC (Mu et al., TDSC 2024)",
    }


__all__ = ["LabelFlippingAttack", "label_flipping_metadata"]
