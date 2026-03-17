from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import Tensor


def extract_bn_features(state_dict: Dict[str, Tensor]) -> Tensor:
    """Extract flattened BN stats (weights, bias, running_mean, running_var)."""

    keys: List[str] = []
    for k in state_dict.keys():
        if "bn" in k and (
            k.endswith("weight")
            or k.endswith("bias")
            or k.endswith("running_mean")
            or k.endswith("running_var")
        ):
            keys.append(k)
    keys.sort()

    feats: List[Tensor] = []
    for k in keys:
        v = state_dict[k].detach().float().view(-1)
        feats.append(v)
    if not feats:
        raise ValueError("No BatchNorm statistics found in state_dict.")
    return torch.cat(feats, dim=0)


def build_bn_matrix(client_state_dicts: Iterable[Dict[str, Tensor]]) -> Tensor:
    """Stack K clients' BN features into shape (K, D_bn)."""

    feat_list: List[Tensor] = [extract_bn_features(sd) for sd in client_state_dicts]
    return torch.stack(feat_list, dim=0)


def robust_scale_features(x: Tensor) -> Tensor:
    """Robust feature-wise scaling using median and MAD.

    This normalizes each BN feature dimension to reduce the influence of outliers
    (e.g., malicious noisy clients) before feeding into the AE/SVDD model.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor for BN features, got {x.ndim}D.")

    med = x.median(dim=0).values
    mad = (x - med).abs().median(dim=0).values
    mad = mad.clamp_min(1e-4)
    return (x - med) / mad


def mad(x: Tensor) -> Tensor:
    """Median Absolute Deviation along dim=0, scaled by 1.4826."""

    med = x.median(dim=0).values
    return 1.4826 * (x - med).abs().median(dim=0).values


def robust_zscore(x: Tensor) -> Tensor:
    """Per-feature robust z-score using median and MAD."""

    med = x.median(dim=0).values
    m = mad(x).clamp(min=1e-8)
    return (x - med) / m


def weighted_fedavg(client_state_dicts: List[Dict[str, Tensor]], alpha: Tensor) -> Dict[str, Tensor]:
    """Weighted FedAvg aggregation over client state_dicts."""

    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts provided.")
    if alpha.ndim != 1 or alpha.numel() != len(client_state_dicts):
        raise ValueError("alpha must be 1D with length K.")

    device = client_state_dicts[0][next(iter(client_state_dicts[0]))].device
    alpha = alpha.to(device)

    keys = client_state_dicts[0].keys()
    agg: Dict[str, Tensor] = {}
    for k in keys:
        stacked = torch.stack([sd[k].to(device) for sd in client_state_dicts], dim=0)
        # reshape alpha for broadcasting
        w = alpha.view(-1, *([1] * (stacked.ndim - 1)))
        agg[k] = (w * stacked).sum(dim=0)
    return agg

