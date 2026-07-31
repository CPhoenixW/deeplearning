from __future__ import annotations

from typing import Any, Dict

from .reporters import get_reporter


def print_round_event(event: Dict[str, Any]) -> None:
    """Render a round using the reporter registered by its defense strategy."""

    get_reporter(str(event["defense"])).print_console(event)


__all__ = ["print_round_event"]
