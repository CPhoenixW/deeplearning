from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class BNReplayBuffer:
    """Replay buffer that stores BN feature vectors across rounds."""

    def __init__(self, capacity: int, d_bn: int) -> None:
        self.capacity = int(capacity)
        self.d_bn = int(d_bn)
        self._storage: Optional[Tensor] = None

    def add(self, X: Tensor) -> None:
        """Add K new BN feature vectors, dropping oldest if over capacity."""

        X = X.detach().cpu()
        if X.ndim != 2 or X.shape[1] != self.d_bn:
            raise ValueError(f"Expected (K, {self.d_bn}) features, got {tuple(X.shape)}")

        if self._storage is None:
            self._storage = X
        else:
            self._storage = torch.cat([self._storage, X], dim=0)

        if self._storage.size(0) > self.capacity:
            overflow = self._storage.size(0) - self.capacity
            self._storage = self._storage[overflow:]

    def sample(self, batch_size: int) -> Tensor:
        """Random sample from buffer. If size < batch_size, return all."""

        if self._storage is None:
            raise RuntimeError("BNReplayBuffer is empty.")

        n = self._storage.size(0)
        if n <= batch_size:
            return self._storage.clone()

        idx = torch.randperm(n)[:batch_size]
        return self._storage[idx]

    def get_all(self) -> Tensor:
        """Return all buffered features."""

        if self._storage is None:
            return torch.empty(0, self.d_bn)
        return self._storage.clone()

    def __len__(self) -> int:
        if self._storage is None:
            return 0
        return int(self._storage.size(0))

