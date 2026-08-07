from __future__ import annotations

import math
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
    *,
    input_mode: str = "absolute",
    reference_state_dict: Dict[str, Tensor] | None = None,
) -> Tensor:
    """Stack per-client SVDD feature rows using a task-specific extractor.

    ``absolute`` extracts features directly from each client model. ``delta``
    first subtracts the pre-round global model from every floating-point entry,
    then applies the same extractor.  Non-floating buffers are copied unchanged;
    the current extractors only consume floating-point parameters/statistics.
    """

    mode = input_mode.lower().strip()
    if mode not in {"absolute", "delta"}:
        raise ValueError(
            f"Unknown SVDD input_mode {input_mode!r}. Use 'absolute' or 'delta'."
        )
    if mode == "delta" and reference_state_dict is None:
        raise ValueError("reference_state_dict is required when input_mode='delta'.")

    feat_list: List[Tensor] = []
    for sd in client_state_dicts:
        feature_sd = sd
        if mode == "delta":
            assert reference_state_dict is not None
            if sd.keys() != reference_state_dict.keys():
                raise ValueError("Client and reference state_dict keys do not match.")
            feature_sd = {}
            for key, client_value in sd.items():
                reference_value = reference_state_dict[key]
                if client_value.shape != reference_value.shape:
                    raise ValueError(
                        f"State shape mismatch for {key!r}: "
                        f"{tuple(client_value.shape)} vs {tuple(reference_value.shape)}."
                    )
                client_cpu = client_value.detach().cpu()
                reference_cpu = reference_value.detach().cpu()
                if client_cpu.is_floating_point():
                    feature_sd[key] = (
                        client_cpu.float() - reference_cpu.float()
                    ).to(dtype=client_cpu.dtype)
                else:
                    feature_sd[key] = client_cpu.clone()
        feat_list.append(extract_fn(feature_sd))
    return torch.stack(feat_list, dim=0)


def robust_scale_features(
    x: Tensor,
    *,
    clip_value: float | None = None,
) -> Tensor:
    """Robust feature-wise scaling using median and MAD.

    This normalizes each BN feature dimension to reduce the influence of outliers
    (e.g., malicious noisy clients) before feeding into the AE/SVDD model.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor for BN features, got {x.ndim}D.")

    safe = torch.nan_to_num(
        x.float(),
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=torch.finfo(torch.float32).min,
    )
    med = safe.median(dim=0).values
    mad = (safe - med).abs().median(dim=0).values
    mad = mad.clamp_min(1e-4)
    scaled = torch.nan_to_num(
        (safe - med) / mad,
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=torch.finfo(torch.float32).min,
    )
    if clip_value is not None:
        clip = float(clip_value)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip_value must be a positive finite number.")
        scaled = scaled.clamp(min=-clip, max=clip)
    return scaled


def mad(x: Tensor) -> Tensor:
    """Median Absolute Deviation along dim=0, scaled by 1.4826."""

    med = x.median(dim=0).values
    return 1.4826 * (x - med).abs().median(dim=0).values


def robust_zscore(x: Tensor) -> Tensor:
    """Per-feature robust z-score using median and MAD."""

    med = x.median(dim=0).values
    m = mad(x).clamp(min=1e-8)
    return (x - med) / m


def _state_dict_is_finite(state_dict: Dict[str, Tensor]) -> bool:
    """Return whether all floating-point state entries are finite.

    Integer buffers (for example ``num_batches_tracked``) are finite by
    construction and do not need a conversion to floating point here.
    """

    for value in state_dict.values():
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all().item()
        ):
            return False
    return True


def weighted_fedavg(
    client_state_dicts: List[Dict[str, Tensor]],
    alpha: Tensor,
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Tensor]:
    """Weighted FedAvg aggregation over client state_dicts.

    Zero-weight clients are omitted before stacking.  This matters for
    filtering defenses: multiplying a rejected ``NaN``/``Inf`` update by zero
    still produces ``NaN`` in PyTorch.  If an active client contains a
    non-finite value, the client is removed and the remaining weights are
    renormalized.  A round with no finite positively weighted client fails
    explicitly instead of silently poisoning the global model.
    """

    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts provided.")
    if alpha.ndim != 1 or alpha.numel() != len(client_state_dicts):
        raise ValueError("alpha must be 1D with length K.")

    aggregation_device = (
        torch.device(device)
        if device is not None
        else client_state_dicts[0][next(iter(client_state_dicts[0]))].device
    )
    alpha = alpha.to(aggregation_device, non_blocking=True)
    if (torch.is_floating_point(alpha) or torch.is_complex(alpha)) and not bool(
        torch.isfinite(alpha).all().item()
    ):
        raise ValueError("alpha must contain only finite values.")
    if bool((alpha < 0).any().item()):
        raise ValueError("alpha must be non-negative.")

    active = alpha > 0
    if not bool(active.any().item()):
        raise ValueError("alpha must assign positive weight to at least one client.")
    active_indices = torch.where(active)[0].detach().cpu().tolist()
    active_states = [client_state_dicts[int(i)] for i in active_indices]
    active_alpha = alpha[active]

    keys = client_state_dicts[0].keys()

    def _aggregate(
        states: List[Dict[str, Tensor]], weights: Tensor
    ) -> tuple[Dict[str, Tensor], bool]:
        result: Dict[str, Tensor] = {}
        finite_flags: List[Tensor] = []
        for key in keys:
            stacked = torch.stack(
                [sd[key].to(aggregation_device, non_blocking=True) for sd in states],
                dim=0,
            )
            # Reshape weights for broadcasting.  ``states`` contains only
            # positively weighted clients, so zero * NaN cannot occur here.
            w = weights.view(-1, *([1] * (stacked.ndim - 1)))
            value = (w * stacked).sum(dim=0)
            if torch.is_floating_point(value) or torch.is_complex(value):
                # Defer the host synchronization until all state entries have
                # been reduced; per-key ``.item()`` calls are costly on CUDA.
                finite_flags.append(torch.isfinite(value).all())
            result[key] = value
        encountered_nonfinite = bool(
            finite_flags and not bool(torch.stack(finite_flags).all().item())
        )
        return result, encountered_nonfinite

    agg, encountered_nonfinite = _aggregate(active_states, active_alpha)
    if not encountered_nonfinite:
        return agg

    # A malicious update can be non-finite in only one parameter tensor.  Find
    # and remove such clients once, then recompute every key with a common
    # finite population so the resulting state remains internally consistent.
    finite_mask = torch.tensor(
        [_state_dict_is_finite(state) for state in active_states],
        dtype=torch.bool,
        device=active_alpha.device,
    )
    if not bool(finite_mask.any().item()):
        raise FloatingPointError("No finite positively weighted client update is available.")
    retained_states = [
        state for state, keep in zip(active_states, finite_mask.detach().cpu().tolist()) if keep
    ]
    retained_alpha = active_alpha[finite_mask]
    weight_sum = retained_alpha.sum()
    if not bool(torch.isfinite(weight_sum).item()) or float(weight_sum.item()) <= 0.0:
        raise FloatingPointError("Finite client weights cannot be normalized.")
    agg, still_nonfinite = _aggregate(retained_states, retained_alpha / weight_sum)
    if still_nonfinite:
        raise FloatingPointError("Finite client aggregation produced a non-finite global state.")
    return agg


def aggregate_fedavg(
    client_state_dicts: List[Dict[str, Tensor]],
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Tensor]:
    """Uniform FedAvg aggregation."""

    if len(client_state_dicts) == 0:
        raise ValueError("No client state_dicts provided.")
    k = len(client_state_dicts)
    alpha = torch.full((k,), 1.0 / k)
    return weighted_fedavg(client_state_dicts, alpha, device=device)


def aggregate_trimmed_mean(
    client_state_dicts: List[Dict[str, Tensor]],
    trim_ratio: float = 0.2,
    num_byzantine: int | None = None,
    *,
    device: torch.device | str | None = None,
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
    aggregation_device = (
        torch.device(device)
        if device is not None
        else client_state_dicts[0][next(iter(client_state_dicts[0]))].device
    )
    agg: Dict[str, Tensor] = {}
    for key in keys:
        tensors = [sd[key] for sd in client_state_dicts]
        ref = tensors[0]
        # Keep non-floating tensors unchanged to preserve dtype (e.g. num_batches_tracked).
        if not ref.is_floating_point():
            agg[key] = ref.detach().to(aggregation_device, non_blocking=True).clone()
            continue

        stacked = torch.stack(
            [t.detach().to(aggregation_device, non_blocking=True).float() for t in tensors],
            dim=0,
        )
        if trim_k == 0:
            agg[key] = stacked.mean(dim=0).to(ref.dtype)
            continue

        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[trim_k : k - trim_k]
        agg[key] = kept.mean(dim=0).to(ref.dtype)
    return agg


def _flatten_floating_params(
    state_dict: Dict[str, Tensor],
    device: torch.device | str | None = None,
) -> Tensor:
    parts: List[Tensor] = []
    for _, value in state_dict.items():
        if value.is_floating_point():
            value = value.detach()
            if device is not None:
                value = value.to(device, non_blocking=True)
            parts.append(value.float().reshape(-1))
    if not parts:
        raise ValueError("No floating-point parameters found in state_dict.")
    return torch.cat(parts, dim=0)


def compute_multi_krum_scores(
    client_state_dicts: List[Dict[str, Tensor]],
    num_byzantine: int,
    *,
    device: torch.device | str | None = None,
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

    updates = torch.stack(
        [_flatten_floating_params(sd, device=device) for sd in client_state_dicts],
        dim=0,
    )  # (n, d)
    sq_norms = (updates * updates).sum(dim=1, keepdim=True)
    distances = sq_norms + sq_norms.t() - 2.0 * (updates @ updates.t())
    distances = distances.clamp_min(0.0)

    neighbors = n - num_byzantine - 2
    scores = torch.empty(n, dtype=distances.dtype, device=distances.device)
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
    *,
    device: torch.device | str | None = None,
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

    scores = compute_multi_krum_scores(
        client_state_dicts,
        num_byzantine=num_byzantine,
        device=device,
    )
    m = n - num_byzantine - 2
    if num_selected is None:
        num_selected = m
    num_selected = max(1, min(num_selected, n))

    selected = torch.topk(scores, k=num_selected, largest=False).indices
    selected_sds = [client_state_dicts[int(idx)] for idx in selected.tolist()]
    return aggregate_fedavg(selected_sds, device=device)


def aggregate_updates(
    client_state_dicts: List[Dict[str, Tensor]],
    method: str,
    *,
    trim_ratio: float = 0.2,
    num_byzantine: int = 0,
    num_selected: int | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, Tensor]:
    """Unified aggregation interface for FedAvg / Trimmed Mean / Multi-Krum."""

    method_norm = method.lower().strip()
    if method_norm == "fedavg":
        return aggregate_fedavg(client_state_dicts, device=device)
    if method_norm == "trimmed_mean":
        return aggregate_trimmed_mean(
            client_state_dicts,
            trim_ratio=trim_ratio,
            num_byzantine=num_byzantine if num_byzantine > 0 else None,
            device=device,
        )
    if method_norm == "multi_krum":
        return aggregate_multi_krum(
            client_state_dicts,
            num_byzantine=num_byzantine,
            num_selected=num_selected,
            device=device,
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
    device: torch.device | str | None = None,
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
        return aggregate_fedavg(client_state_dicts, device=device), torch.ones(n), alpha

    if method_norm == "trimmed_mean":
        # Coordinate-wise trimmed mean has no unique client-level reject mask.
        alpha = torch.full((n,), 1.0 / n)
        return (
            aggregate_trimmed_mean(
                client_state_dicts,
                trim_ratio=trim_ratio,
                num_byzantine=num_byzantine if num_byzantine > 0 else None,
                device=device,
            ),
            torch.ones(n),
            alpha,
        )

    if method_norm == "multi_krum":
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be >= 0.")
        if n <= 2 * num_byzantine + 2:
            raise ValueError("Multi-Krum requires n > 2 * num_byzantine + 2.")

        scores = compute_multi_krum_scores(
            client_state_dicts,
            num_byzantine=num_byzantine,
            device=device,
        )
        m = n - num_byzantine - 2
        if num_selected is None:
            num_selected = m
        num_selected = max(1, min(num_selected, n))

        selected = torch.topk(scores, k=num_selected, largest=False).indices
        selected_cpu = selected.detach().cpu()
        m_mask = torch.zeros(n)
        m_mask[selected_cpu] = 1.0
        alpha = m_mask / m_mask.sum()

        selected_sds = [client_state_dicts[int(idx)] for idx in selected_cpu.tolist()]
        global_sd = aggregate_fedavg(selected_sds, device=device)
        return global_sd, m_mask, alpha

    raise ValueError(
        f"Unknown aggregation method: {method}. "
        "Expected one of ['fedavg', 'trimmed_mean', 'multi_krum']."
    )
