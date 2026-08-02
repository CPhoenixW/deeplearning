"""Shared attack-client interfaces."""

from __future__ import annotations

from ..clients import BenignClient


class MaliciousClient(BenignClient):
    """Base class for clients that alter local data or their uploaded model."""


__all__ = ["MaliciousClient"]
