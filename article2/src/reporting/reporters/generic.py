from __future__ import annotations

from typing import Any, Dict

from .base import RoundReporter


class GenericReporter(RoundReporter):
    title = "Federated Aggregation"
    metric_key = "score"
    metric_label = "Score"
