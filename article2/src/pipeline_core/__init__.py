"""Composable execution stages for federated experiments."""

from .contracts import PipelineContext, Stage
from .result import StructuredResultWriter

__all__ = ["PipelineContext", "Stage", "StructuredResultWriter"]
