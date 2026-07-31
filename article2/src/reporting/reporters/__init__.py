from __future__ import annotations

from typing import Dict, Type

from .base import RoundReporter
from .alignins import AlignInsReporter
from .bnguard import BNGuardReporter
from .fedseca import FedSECAReporter
from .fedavg import FedAvgReporter
from .fl_defender import FLDefenderReporter
from .flanders import FLANDERSReporter
from .flgmm import FLGMMReporter
from .generic import GenericReporter
from .lasa import LASAReporter
from .multi_krum import MultiKrumReporter
from .svdd import SVDDReporter
from .trimmed_mean import TrimmedMeanReporter


REPORTER_REGISTRY: Dict[str, Type[RoundReporter]] = {
    "avg": FedAvgReporter,
    "tm": TrimmedMeanReporter,
    "mk": MultiKrumReporter,
    "svdd": SVDDReporter,
    "dmc": GenericReporter,
    "lasa": LASAReporter,
    "seca": FedSECAReporter,
    "fld": FLDefenderReporter,
    "alignins": AlignInsReporter,
    "bnguard": BNGuardReporter,
    "flgmm": FLGMMReporter,
    "flanders": FLANDERSReporter,
}


def get_reporter(defense_name: str) -> RoundReporter:
    return REPORTER_REGISTRY.get(defense_name.lower().strip(), GenericReporter)()


__all__ = ["RoundReporter", "get_reporter", "REPORTER_REGISTRY"]
