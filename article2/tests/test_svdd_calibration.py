from __future__ import annotations

import copy

import pytest

from tools.svdd_calibration import build_candidates, promote_manifest


def _manifest() -> dict:
    return {
        "task": "cifar10",
        "attacks": ["none", "mix"],
        "seeds": [42],
        "num_candidates": 8,
        "design_seed": 9,
        "baseline_parameters": {
            "param_descriptor_dim": 4096,
            "latent_dim": 64,
            "phase1_rounds": 15,
            "ae_warmup_keep_ratio": 0.8,
            "ae_lr": 0.001,
            "tau_start": 3.0,
            "tau_end": 2.0,
            "tau_anneal_rounds": 100,
            "center_ema_decay": 0.9,
            "svdd_loss_weight": 1.0,
            "recon_loss_weight": 1.0,
            "svdd_feature_clip": 10.0,
        },
        "parameter_space": {
            "param_descriptor_dim": [256, 4096],
            "latent_dim": [8, 64],
            "phase1_rounds": [5, 15],
            "ae_warmup_keep_ratio": [0.6, 0.8],
            "ae_lr": [0.0001, 0.001],
            "tau_start": [2.5, 3.0],
            "tau_end": [2.0, 3.0],
            "tau_anneal_rounds": [30, 100],
            "center_ema_decay": [0.5, 0.9],
            "loss_weight_ratio": [0.5, 1.0],
            "svdd_feature_clip": [5.0, 10.0],
        },
        "promotion_defaults": {
            "top_k": 2,
            "total_rounds": 300,
            "seeds": [42, 43, 44],
        },
    }


def test_balanced_candidates_are_deterministic_unique_and_valid() -> None:
    manifest = _manifest()
    first = build_candidates(manifest)
    second = build_candidates(copy.deepcopy(manifest))
    assert first == second
    assert len(first) == 8
    assert len({str(sorted(item.items())) for item in first}) == 8
    assert first[0] == manifest["baseline_parameters"]
    assert all(item["tau_end"] <= item["tau_start"] for item in first)
    assert all(item["recon_loss_weight"] == 1.0 for item in first)


def test_candidate_validation_rejects_increasing_tau() -> None:
    manifest = _manifest()
    manifest["baseline_parameters"]["tau_start"] = 2.5
    manifest["baseline_parameters"]["tau_end"] = 3.0
    with pytest.raises(ValueError, match="tau_end"):
        build_candidates(manifest)


def test_promotion_uses_ranked_explicit_candidates_and_full_budget() -> None:
    manifest = _manifest()
    candidates = build_candidates(manifest)
    selection = {
        "rankings": [
            {"parameters": candidates[3]},
            {"parameters": candidates[1]},
        ]
    }
    promoted = promote_manifest(manifest, selection)
    assert promoted["explicit_candidates"] == [candidates[3], candidates[1]]
    assert promoted["seeds"] == [42, 43, 44]
    assert promoted["common_overrides"]["total_rounds"] == 300
