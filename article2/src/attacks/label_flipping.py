"""Label-flipping data-poisoning attack."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor

from .base import MaliciousClient


class LabelFlippingAttack(MaliciousClient):
    """Train with the symmetric label mapping ``y' = C - 1 - y``."""

    def _transform_batch(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        classes = int(self.config.num_classes)
        return x, (classes - 1) - y


__all__ = ["LabelFlippingAttack"]
