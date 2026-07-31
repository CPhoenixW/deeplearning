from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def _iter_floating_items_in_order(state_dict: Dict[str, Tensor]) -> List[tuple[str, Tensor]]:
    """Deterministic floating-parameter iteration for flatten/unflatten.

    We follow the native `state_dict()` insertion order to stay consistent across clients.
    """
    items: List[tuple[str, Tensor]] = []
    for k, v in state_dict.items():
        if v.is_floating_point():
            items.append((k, v))
    if not items:
        raise ValueError("No floating-point tensors found in state_dict.")
    return items


def _flatten_delta(global_sd: Dict[str, Tensor], client_sd: Dict[str, Tensor]) -> Tensor:
    parts: List[Tensor] = []
    for k, g in _iter_floating_items_in_order(global_sd):
        c = client_sd[k]
        parts.append((c.detach().cpu().float() - g.detach().cpu().float()).reshape(-1))
    return torch.cat(parts, dim=0)


def _flatten_param_delta(
    global_sd: Dict[str, Tensor],
    client_sd: Dict[str, Tensor],
    param_names: List[str],
) -> Tensor:
    parts: List[Tensor] = []
    for name in param_names:
        g = global_sd[name].detach().cpu()
        c = client_sd[name].detach().cpu()
        parts.append((c.float() - g.float()).reshape(-1))
    if not parts:
        raise ValueError("No parameters found to flatten.")
    return torch.cat(parts, dim=0)


def _apply_flat_param_delta_to_global(
    global_sd: Dict[str, Tensor],
    delta_flat: Tensor,
    param_names: List[str],
) -> Dict[str, Tensor]:
    """Apply delta_flat to *parameters only*, keep all buffers from global_sd unchanged."""
    out: Dict[str, Tensor] = {k: v.detach().cpu().clone() for k, v in global_sd.items()}
    i = 0
    delta_flat_cpu = delta_flat.detach().cpu()
    for name in param_names:
        g = global_sd[name].detach().cpu()
        numel = int(g.numel())
        d = delta_flat_cpu[i : i + numel].view(g.shape).to(dtype=g.dtype)
        out[name] = (g + d).clone()
        i += numel
    if i != int(delta_flat_cpu.numel()):
        raise ValueError(f"Delta length mismatch: used {i}, total {int(delta_flat_cpu.numel())}.")
    return out


def _aggregate_nonparam_buffers(
    global_sd: Dict[str, Tensor],
    client_sds: List[Dict[str, Tensor]],
    param_names: List[str],
    client_weights: Tensor,
) -> Dict[str, Tensor]:
    """Aggregate non-parameter entries (e.g., BN running stats) from selected clients.

    We keep robust-aggregation behavior on trainable parameters, while still updating
    model buffers that are required for stable evaluation with BatchNorm.
    """
    if len(client_sds) == 0:
        return {k: v.detach().cpu().clone() for k, v in global_sd.items()}

    n = len(client_sds)
    w = client_weights.detach().cpu().float().view(-1)
    if w.numel() != n:
        raise ValueError(f"client_weights length mismatch: got {w.numel()}, expected {n}.")
    w_sum = float(w.sum().item())
    if w_sum <= 0.0:
        w = torch.full((n,), 1.0 / float(n), dtype=torch.float32)
    else:
        w = w / w_sum

    param_set = set(param_names)
    out: Dict[str, Tensor] = {}
    for k, g in global_sd.items():
        if k in param_set:
            out[k] = g.detach().cpu().clone()
            continue
        if g.is_floating_point():
            stacked = torch.stack([sd[k].detach().cpu().float() for sd in client_sds], dim=0)
            wk = w.view(-1, *([1] * (stacked.ndim - 1)))
            out[k] = (wk * stacked).sum(dim=0).to(dtype=g.dtype).clone()
        else:
            # Keep integer buffers deterministic; use the top-weight client snapshot.
            top_idx = int(torch.argmax(w).item())
            out[k] = client_sds[top_idx][k].detach().cpu().clone()
    return out


def _apply_flat_delta_to_global(
    global_sd: Dict[str, Tensor],
    delta_flat: Tensor,
) -> Dict[str, Tensor]:
    """Return new global state_dict = global + unflatten(delta_flat) for floating tensors."""
    out: Dict[str, Tensor] = {}
    i = 0
    delta_flat_cpu = delta_flat.detach().cpu()
    for k, v in global_sd.items():
        g = v.detach().cpu()
        if not g.is_floating_point():
            out[k] = g.clone()
            continue
        numel = g.numel()
        d = delta_flat_cpu[i : i + numel].view(g.shape).to(dtype=g.dtype)
        out[k] = (g + d).clone()
        i += numel
    if i != int(delta_flat_cpu.numel()):
        raise ValueError(f"Delta length mismatch: used {i}, total {int(delta_flat_cpu.numel())}.")
    return out


def _lasa_layer_dims_from_model(model: nn.Module, param_names: List[str]) -> np.ndarray:
    """Build LASA layer boundaries that are always compatible with flattened parameter deltas.

    For CNN-style models, we keep the original LASA grouping over (BN/Linear/Conv) layers.
    For models with additional parameter types (e.g., Embedding/Transformer/LayerNorm),
    we fall back to parameter-wise grouping so the concatenated LASA delta length exactly
    matches `_flatten_param_delta(..., param_names)`.
    """
    dims: List[int] = [0]
    for layer in model.modules():
        if isinstance(layer, (nn.BatchNorm2d, nn.Linear, nn.Conv2d)):
            layer_dims = int(layer.weight.numel())
            if layer.bias is not None:
                layer_dims += int(layer.bias.numel())
            dims.append(layer_dims)
    layer_dims = np.cumsum(np.array(dims, dtype=np.int64))

    total_param_numel = int(sum(int(p.numel()) for _name, p in model.named_parameters()))
    if layer_dims.size > 0 and int(layer_dims[-1]) == total_param_numel:
        return layer_dims

    # Fallback: parameter-wise boundaries in the same order used by flatten/apply helpers.
    param_sizes: List[int] = [0]
    named_params = dict(model.named_parameters())
    for name in param_names:
        p = named_params.get(name)
        if p is None:
            raise KeyError(f"Parameter {name!r} not found in model.named_parameters().")
        param_sizes.append(int(p.numel()))
    return np.cumsum(np.array(param_sizes, dtype=np.int64))


def _topk_sparsification(vector: Tensor, sparsity_ratio: float) -> Tensor:
    k_dim = int(float(sparsity_ratio) * int(vector.numel()))
    k_dim = max(0, min(k_dim, int(vector.numel())))
    if k_dim == 0:
        return torch.zeros_like(vector)
    sign_vec = vector.sign()
    sparse_update = torch.zeros_like(vector)
    vals, indices = torch.topk(vector.abs(), k_dim)
    sparse_update[indices] = vals
    sparse_update *= sign_vec
    return sparse_update

def _cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    # x: (n, d)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_unit = x / x_norm
    return x_unit @ x_unit.T


def _standardize_matrix(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=0, keepdims=True)
    sigma = m.std(axis=0, keepdims=True) + 1e-12
    return (m - mu) / sigma


def _pca_2d_via_svd(m: np.ndarray, n_components: int) -> np.ndarray:
    # Standard PCA on rows: project m onto top principal components.
    # Assumes m already standardized if needed.
    n_components = int(max(1, min(n_components, min(m.shape[0], m.shape[1]))))
    m0 = m - m.mean(axis=0, keepdims=True)
    # SVD: m0 = U S Vt, principal directions are rows of Vt
    _u, _s, vt = np.linalg.svd(m0, full_matrices=False)
    components = vt[:n_components].T  # (p, n_components)
    return m0 @ components  # (n, n_components)


def _mad_zscores(values: Tensor) -> Tensor:
    """Per-client robust z-scores using median and MAD (dim=0 over the batch)."""

    med = values.median()
    mad = (values - med).abs().median()
    mad = torch.clamp(mad, min=1e-6)
    return (values - med).abs() / mad


def _state_l2_distance(left: Dict[str, Tensor], right: Dict[str, Tensor]) -> float:
    dist_sq = 0.0
    for k, l in left.items():
        if not l.is_floating_point():
            continue
        r = right[k]
        diff = l.detach().cpu().float() - r.detach().cpu().float()
        dist_sq += float((diff * diff).sum().item())
    return float(np.sqrt(max(dist_sq, 0.0)))


def _fit_1d_gmm_largest_cluster(
    values: np.ndarray,
    *,
    n_iter: int,
) -> Tuple[np.ndarray, float, float]:
    """Fit a tiny two-component 1-D GMM and return the largest component mask.

    FLGMM's official code uses sklearn GaussianMixture.  This EM routine keeps
    the same modeling idea without adding sklearn as a runtime dependency.
    """

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0, 1.0
    if n < 2 or float(np.std(x)) < 1e-12:
        return np.ones(n, dtype=bool), float(np.mean(x)), max(float(np.std(x)), 1e-6)

    means = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)], dtype=np.float64)
    variances = np.full(2, max(float(np.var(x)), 1e-6), dtype=np.float64)
    weights = np.full(2, 0.5, dtype=np.float64)

    for _ in range(max(1, int(n_iter))):
        probs = []
        for j in range(2):
            var = max(float(variances[j]), 1e-6)
            coef = weights[j] / np.sqrt(2.0 * np.pi * var)
            probs.append(coef * np.exp(-0.5 * ((x - means[j]) ** 2) / var))
        resp = np.stack(probs, axis=1)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / np.clip(resp_sum, 1e-12, None)
        nk = resp.sum(axis=0)
        for j in range(2):
            if nk[j] <= 1e-12:
                continue
            weights[j] = nk[j] / float(n)
            means[j] = float((resp[:, j] * x).sum() / nk[j])
            variances[j] = float((resp[:, j] * ((x - means[j]) ** 2)).sum() / nk[j])
            variances[j] = max(float(variances[j]), 1e-6)

    labels = np.argmax(resp, axis=1)
    counts = np.bincount(labels, minlength=2)
    normal_component = int(np.argmax(counts))
    mask = labels == normal_component
    normal_vals = x[mask]
    return mask, float(np.mean(normal_vals)), max(float(np.std(normal_vals)), 1e-6)


def _sample_flat_delta(
    global_sd: Dict[str, Tensor],
    client_sd: Dict[str, Tensor],
    indices: Optional[Tensor],
) -> Tensor:
    flat = _flatten_delta(global_sd, client_sd)
    if indices is None:
        return flat
    return flat[indices]


def _mar_forecast(
    X: np.ndarray,
    *,
    pred_step: int,
    alpha: float,
    beta: float,
    maxiter: int,
) -> np.ndarray:
    """Matrix autoregressive forecast used by FLANDERS.

    X shape is (clients, sampled_params, time).  The implementation follows the
    Flower baseline's MAR routine, with pseudoinverse for numerical stability.
    """

    m, n, T = X.shape
    if T < 2:
        return X[:, :, -1:][:, :, :pred_step]

    rng = np.random.default_rng(0)
    A = rng.standard_normal((m, m))
    B = rng.standard_normal((n, n))
    x_min = float(np.min(X))
    x_scale = float(np.max(X) - x_min)
    if abs(x_scale) < 1e-12:
        X_norm = np.zeros_like(X, dtype=np.float64)
    else:
        X_norm = (X - x_min) / x_scale

    eye_m = np.identity(m)
    eye_n = np.identity(n)
    for _ in range(max(1, int(maxiter))):
        bt_b = B.T @ B
        lhs = np.zeros((m, m))
        rhs = np.zeros((m, m))
        for t in range(1, T):
            lhs += X_norm[:, :, t] @ B @ X_norm[:, :, t - 1].T
            rhs += X_norm[:, :, t - 1] @ bt_b @ X_norm[:, :, t - 1].T
        A = lhs @ np.linalg.pinv(rhs + float(alpha) * eye_m)

        at_a = A.T @ A
        lhs = np.zeros((n, n))
        rhs = np.zeros((n, n))
        for t in range(1, T):
            lhs += X_norm[:, :, t].T @ A @ X_norm[:, :, t - 1]
            rhs += X_norm[:, :, t - 1].T @ at_a @ X_norm[:, :, t - 1]
        B = lhs @ np.linalg.pinv(rhs + float(beta) * eye_n)

    tensor = np.append(X, np.zeros((m, n, pred_step), dtype=np.float64), axis=2)
    for s in range(pred_step):
        tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
    return tensor[:, :, -pred_step:]



