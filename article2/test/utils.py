from __future__ import annotations

from typing import Callable, Dict, Iterable, List

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


def extract_transformer_encoder_layernorm_features(state_dict: Dict[str, Tensor]) -> Tensor:
    """Flatten LayerNorm gamma/beta from ``nn.TransformerEncoder`` submodules (norm1/norm2 per layer)."""

    keys: List[str] = []
    for k in state_dict.keys():
        if not (k.endswith("weight") or k.endswith("bias")):
            continue
        if "encoder.layers" not in k:
            continue
        if ".norm1." not in k and ".norm2." not in k:
            continue
        keys.append(k)
    keys.sort()

    feats: List[Tensor] = []
    for k in keys:
        v = state_dict[k].detach().float().view(-1)
        feats.append(v)
    if not feats:
        raise ValueError("No TransformerEncoder LayerNorm params found in state_dict.")
    return torch.cat(feats, dim=0)


def extract_ag_news_svdd_features(state_dict: Dict[str, Tensor], mode: str) -> Tensor:
    """SVDD feature vector for AG News text model (see ``FedConfig.ag_news_svdd_features``)."""

    m = mode.lower().strip()
    if m == "bn":
        return extract_bn_features(state_dict)
    if m == "ln":
        return extract_transformer_encoder_layernorm_features(state_dict)
    if m == "ln_bn":
        ln = extract_transformer_encoder_layernorm_features(state_dict)
        bn = extract_bn_features(state_dict)
        return torch.cat([ln, bn], dim=0)
    raise ValueError(
        f"Unknown ag_news_svdd_features mode {mode!r}. Use 'bn', 'ln', or 'ln_bn'."
    )


def build_bn_matrix(client_state_dicts: Iterable[Dict[str, Tensor]]) -> Tensor:
    """Stack K clients' BN features into shape (K, D_bn)."""

    feat_list: List[Tensor] = [extract_bn_features(sd) for sd in client_state_dicts]
    return torch.stack(feat_list, dim=0)


def build_svdd_feature_matrix(
    client_state_dicts: Iterable[Dict[str, Tensor]],
    extract_fn: Callable[[Dict[str, Tensor]], Tensor],
) -> Tensor:
    """Stack per-client SVDD feature rows using a task-specific extractor."""

    feat_list: List[Tensor] = [extract_fn(sd) for sd in client_state_dicts]
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


def aggregate_fedavg(client_state_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Uniform FedAvg aggregation."""

    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts provided.")
    k = len(client_state_dicts)
    alpha = torch.full((k,), 1.0 / k)
    return weighted_fedavg(client_state_dicts, alpha)


def aggregate_trimmed_mean(
    client_state_dicts: List[Dict[str, Tensor]],
    trim_ratio: float = 0.2,
    num_byzantine: int | None = None,
) -> Dict[str, Tensor]:
    """Coordinate-wise Trimmed Mean aggregation.

    Paper-consistent mode:
      remove the largest `b` and smallest `b` values per coordinate, where `b=num_byzantine`.
    Backward-compatible mode:
      when `num_byzantine is None`, use `trim_ratio` to infer trim count.
    """

    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts provided.")
    k = len(client_state_dicts)
    if num_byzantine is None:
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0.0, 0.5).")
        trim_k = int(k * trim_ratio)
    else:
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be >= 0.")
        if 2 * num_byzantine >= k:
            raise ValueError(
                f"Trimmed Mean requires 2*b < n. Got b={num_byzantine}, n={k}."
            )
        trim_k = int(num_byzantine)

    keys = client_state_dicts[0].keys()
    agg: Dict[str, Tensor] = {}
    for key in keys:
        tensors = [sd[key] for sd in client_state_dicts]
        ref = tensors[0]
        # Keep non-floating tensors unchanged to preserve dtype (e.g. num_batches_tracked).
        if not ref.is_floating_point():
            agg[key] = ref.detach().clone()
            continue

        stacked = torch.stack([t.detach().float() for t in tensors], dim=0)
        if trim_k == 0:
            agg[key] = stacked.mean(dim=0).to(ref.dtype)
            continue

        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[trim_k : k - trim_k]
        agg[key] = kept.mean(dim=0).to(ref.dtype)
    return agg


def _flatten_floating_params(state_dict: Dict[str, Tensor]) -> Tensor:
    parts: List[Tensor] = []
    for _, value in state_dict.items():
        if value.is_floating_point():
            parts.append(value.detach().float().reshape(-1))
    if not parts:
        raise ValueError("No floating-point parameters found in state_dict.")
    return torch.cat(parts, dim=0)


def compute_multi_krum_scores(
    client_state_dicts: List[Dict[str, Tensor]],
    num_byzantine: int,
) -> Tensor:
    """Compute per-client Multi-Krum scores.

    score(i) = sum of squared distances to i's nearest (n - f - 2) neighbors.
    """
    n = len(client_state_dicts)
    if n == 0:
        raise ValueError("No client state_dicts provided.")
    if num_byzantine < 0:
        raise ValueError("num_byzantine must be >= 0.")
    if n <= 2 * num_byzantine + 2:
        raise ValueError("Multi-Krum requires n > 2 * num_byzantine + 2.")

    updates = torch.stack([_flatten_floating_params(sd) for sd in client_state_dicts], dim=0)  # (n, d)
    sq_norms = (updates * updates).sum(dim=1, keepdim=True)
    distances = sq_norms + sq_norms.t() - 2.0 * (updates @ updates.t())
    distances = distances.clamp_min(0.0)

    neighbors = n - num_byzantine - 2
    scores = torch.empty(n, dtype=distances.dtype)
    for i in range(n):
        d_i = distances[i]
        others = torch.cat([d_i[:i], d_i[i + 1 :]], dim=0)
        nearest, _ = torch.topk(others, k=neighbors, largest=False)
        scores[i] = nearest.sum()
    return scores


def aggregate_multi_krum(
    client_state_dicts: List[Dict[str, Tensor]],
    num_byzantine: int,
    num_selected: int | None = None,
) -> Dict[str, Tensor]:
    """Multi-Krum aggregation.

    Select `num_selected` clients with the smallest Krum scores and return
    their uniform average.
    """

    n = len(client_state_dicts)
    if n == 0:
        raise ValueError("No client state_dicts provided.")
    if num_byzantine < 0:
        raise ValueError("num_byzantine must be >= 0.")
    if n <= 2 * num_byzantine + 2:
        raise ValueError("Multi-Krum requires n > 2 * num_byzantine + 2.")

    scores = compute_multi_krum_scores(client_state_dicts, num_byzantine=num_byzantine)
    m = n - num_byzantine - 2
    if num_selected is None:
        num_selected = m
    num_selected = max(1, min(num_selected, n))

    selected = torch.topk(scores, k=num_selected, largest=False).indices
    selected_sds = [client_state_dicts[int(idx)] for idx in selected.tolist()]
    return aggregate_fedavg(selected_sds)


def aggregate_updates(
    client_state_dicts: List[Dict[str, Tensor]],
    method: str,
    *,
    trim_ratio: float = 0.2,
    num_byzantine: int = 0,
    num_selected: int | None = None,
) -> Dict[str, Tensor]:
    """Unified aggregation interface for FedAvg / Trimmed Mean / Multi-Krum."""

    method_norm = method.lower().strip()
    if method_norm == "fedavg":
        return aggregate_fedavg(client_state_dicts)
    if method_norm == "trimmed_mean":
        return aggregate_trimmed_mean(
            client_state_dicts,
            trim_ratio=trim_ratio,
            num_byzantine=num_byzantine if num_byzantine > 0 else None,
        )
    if method_norm == "multi_krum":
        return aggregate_multi_krum(
            client_state_dicts,
            num_byzantine=num_byzantine,
            num_selected=num_selected,
        )
    raise ValueError(
        f"Unknown aggregation method: {method}. "
        "Expected one of ['fedavg', 'trimmed_mean', 'multi_krum']."
    )


def aggregate_updates_with_info(
    client_state_dicts: List[Dict[str, Tensor]],
    method: str,
    *,
    trim_ratio: float = 0.2,
    num_byzantine: int = 0,
    num_selected: int | None = None,
) -> tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Aggregation with client-level participation info.

    Returns:
        global_state_dict, M, alpha
        - M: 1 means selected/kept, 0 means rejected (client-level)
        - alpha: client aggregation weights
    """

    n = len(client_state_dicts)
    if n == 0:
        raise ValueError("No client state_dicts provided.")

    method_norm = method.lower().strip()
    if method_norm == "fedavg":
        alpha = torch.full((n,), 1.0 / n)
        return aggregate_fedavg(client_state_dicts), torch.ones(n), alpha

    if method_norm == "trimmed_mean":
        # Coordinate-wise trimmed mean has no unique client-level reject mask.
        alpha = torch.full((n,), 1.0 / n)
        return (
            aggregate_trimmed_mean(
                client_state_dicts,
                trim_ratio=trim_ratio,
                num_byzantine=num_byzantine if num_byzantine > 0 else None,
            ),
            torch.ones(n),
            alpha,
        )

    if method_norm == "multi_krum":
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be >= 0.")
        if n <= 2 * num_byzantine + 2:
            raise ValueError("Multi-Krum requires n > 2 * num_byzantine + 2.")

        scores = compute_multi_krum_scores(client_state_dicts, num_byzantine=num_byzantine)
        m = n - num_byzantine - 2
        if num_selected is None:
            num_selected = m
        num_selected = max(1, min(num_selected, n))

        selected = torch.topk(scores, k=num_selected, largest=False).indices
        m_mask = torch.zeros(n)
        m_mask[selected] = 1.0
        alpha = m_mask / m_mask.sum()

        selected_sds = [client_state_dicts[int(idx)] for idx in selected.tolist()]
        global_sd = aggregate_fedavg(selected_sds)
        return global_sd, m_mask, alpha

    raise ValueError(
        f"Unknown aggregation method: {method}. "
        "Expected one of ['fedavg', 'trimmed_mean', 'multi_krum']."
    )

