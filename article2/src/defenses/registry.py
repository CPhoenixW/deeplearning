from __future__ import annotations

from typing import Dict, Type

from .alignins import AlignInsDefense
from .bnguard import BNGuardDefense
from .fedavg import FedAvgDefense
from .fedseca import FedSECADefense
from .fl_defender import FLDefenderDefense
from .flanders import FLANDERSDefense
from .flgmm import FLGMMDefense
from .lasa import LASADefense
from .multi_krum import MultiKrumDefense
from .svdd import SVDDDefense
from .dmc import DMCDefense
from .trimmed_mean import TrimmedMeanDefense


DEFENSE_REGISTRY: Dict[str, Type] = {
    "avg": FedAvgDefense,
    "tm": TrimmedMeanDefense,
    "mk": MultiKrumDefense,
    "svdd": SVDDDefense,
    "dmc": DMCDefense,
    "lasa": LASADefense,
    "seca": FedSECADefense,
    "fld": FLDefenderDefense,
    "alignins": AlignInsDefense,
    "bnguard": BNGuardDefense,
    "flgmm": FLGMMDefense,
    "flanders": FLANDERSDefense,
}


def create_defense(name: str, *args, **kwargs):
    try:
        defense_cls = DEFENSE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown defense {name!r}; available: {sorted(DEFENSE_REGISTRY)}") from exc
    return defense_cls(*args, **kwargs)
