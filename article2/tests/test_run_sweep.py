from __future__ import annotations

from src.run_sweep import _alpha, _auc, _auprc


def test_alpha_parser_supports_iid_and_dirichlet() -> None:
    assert _alpha("iid") is None
    assert _alpha("0.5") == 0.5


def test_auc_is_tie_aware_and_returns_none_for_single_class() -> None:
    assert _auc([0, 1], [0.1, 0.9]) == 1.0
    assert _auc([0, 1], [0.5, 0.5]) == 0.5
    assert _auc([1, 1], [0.1, 0.2]) is None
    assert _auprc([0, 1], [0.1, 0.9]) == 1.0
    assert _auprc([1, 1], [0.1, 0.2]) == 1.0
