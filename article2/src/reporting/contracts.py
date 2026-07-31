from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class ParticipantResult:
    client_id: int
    role: str
    accepted: bool
    weight: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class DefenseReporter(Protocol):
    def build_round(self, event: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def print_console(self, event: Dict[str, Any]) -> None:
        ...


__all__ = ["DefenseReporter", "ParticipantResult"]
