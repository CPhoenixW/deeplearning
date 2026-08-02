from __future__ import annotations

import pytest
import torch

from src.config import FedConfig
from src.defenses.svdd import _compute_svdd_keep_mask


def _scheduled_tau(config: FedConfig, round_index: int) -> float:
    _mask, tau, _threshold = _compute_svdd_keep_mask(
        torch.tensor([0.5, 1.0, 1.5]),
        round_index,
        config,
    )
    return tau


def test_tau_schedule_uses_literal_start_and_end_rounds() -> None:
    config = FedConfig(
        tau_start=4.0,
        tau_end=2.0,
        tau_anneal_rounds=3,
    )
    assert _scheduled_tau(config, 1) == pytest.approx(4.0)
    assert _scheduled_tau(config, 2) == pytest.approx(3.0)
    assert _scheduled_tau(config, 3) == pytest.approx(2.0)
    assert _scheduled_tau(config, 20) == pytest.approx(2.0)


def test_deprecated_svdd_warmup_rounds_alias_remains_reproducible() -> None:
    config = FedConfig(
        tau_start=3.0,
        tau_end=1.0,
        tau_anneal_rounds=100,
        svdd_warmup_rounds=2,
    )
    assert _scheduled_tau(config, 1) == pytest.approx(3.0)
    assert _scheduled_tau(config, 2) == pytest.approx(1.0)


def test_tau_schedule_rejects_an_increasing_threshold_multiplier() -> None:
    config = FedConfig(tau_start=2.5, tau_end=3.0)
    with pytest.raises(ValueError, match="tau_end"):
        _scheduled_tau(config, 1)
