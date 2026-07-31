"""Defense-owned console and structured result reporting."""

from .console import print_round_event
from .contracts import DefenseReporter, ParticipantResult
from .reporters import REPORTER_REGISTRY, get_reporter

__all__ = [
    "DefenseReporter",
    "ParticipantResult",
    "REPORTER_REGISTRY",
    "get_reporter",
    "print_round_event",
]
