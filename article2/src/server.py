#MinMax's version of server.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

try:
    from .config import FedConfig
    from .models import AutoEncoder
    from .utils import (
        aggregate_updates_with_info,
        build_svdd_feature_matrix,
        compute_multi_krum_scores,
        extract_bn_features,
        robust_scale_features,
        weighted_fedavg,
    )
except ImportError:
    from config import FedConfig
    from models import AutoEncoder
    from utils import (
        aggregate_updates_with_info,
        build_svdd_feature_matrix,
        compute_multi_krum_scores,
        extract_bn_features,
        robust_scale_features,
        weighted_fedavg,
    )


@dataclass
class RoundStats:
    center_norm: float
    z_var: float
    ae_loss: float
    svdd_loss: float
    d: Tensor
    m: Tensor
    alpha: Tensor
    phase: str
    show_detection: bool
    monitor_items: List[Tuple[str, str]]


class BaseServer:
    """Base server class for federated aggregation defenses."""

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        self.config = config
        self.device = device
        self.d_bn = d_bn

        self.global_model = model_fn().to(self.device)
        # Parameter key order used by several ported SOTA aggregators.
        # We intentionally exclude buffers (e.g., BN running stats) to mirror the
        # original designs which operate on trainable parameters / gradients.
        self.param_names: List[str] = [name for name, _p in self.global_model.named_parameters()]

    def state_dict_for_clients(self) -> Dict[str, Tensor]:
        sd = self.global_model.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        raise NotImplementedError


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


class LASAServer(BaseServer):
    """Port of LASA (WACV 2025) from experiment/FL-Byzantine-Library/aggregators/lasa.py.

    Operates on flattened client deltas: Δ_k = w_k - w_g.
    """

    defense_name = "lasa"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self.layer_dims = _lasa_layer_dims_from_model(self.global_model, self.param_names)

    def _gradient_sanitization_and_clipping(self, updates: Tensor) -> Tensor:
        # updates: (K, D)
        # Mirror library behavior: median-norm clipping.
        grad_norm = torch.norm(updates, dim=1, keepdim=True)  # (K,1)
        norm_clip = torch.median(grad_norm, dim=0).values.item()
        grad_norm_clipped = torch.clamp(grad_norm, 0.0, float(norm_clip))
        grads_clip = (updates / (grad_norm + 1e-12)) * grad_norm_clipped
        return grads_clip

    def _byzantine_detection_layer(self, sparse_updates: List[Tensor], start_dim: int, end_dim: int) -> List[int]:
        all_set = set(range(len(sparse_updates)))
        layer_flat_params: List[Tensor] = [u[start_dim:end_dim] for u in sparse_updates]
        if len(layer_flat_params) == 0:
            return list(all_set)
        grads = torch.stack(layer_flat_params, dim=0)

        # Norm check (MZ-score via median/std)
        grad_l2norm = torch.norm(grads.float(), dim=1).detach().cpu().numpy()
        norm_med = float(np.median(grad_l2norm))
        norm_std = float(np.std(grad_l2norm))
        norm_scores = []
        for v in grad_l2norm:
            score = float(abs((float(v) - norm_med) / norm_std)) if norm_std > 0 else 0.0
            norm_scores.append(score)
        benign_idx1 = all_set.intersection(set(np.argwhere(np.array(norm_scores) < float(self.config.lasa_lambda_n)).flatten().astype(int).tolist()))

        # Sign check
        layer_signs: List[float] = []
        for layer_param in layer_flat_params:
            sign_sum = torch.sum(torch.sign(layer_param))
            abs_sign_sum = torch.sum(torch.abs(torch.sign(layer_param)))
            if abs_sign_sum > 0:
                sign_ratio = 0.5 * (1 + sign_sum / abs_sign_sum * (1 - float(self.config.lasa_sparsity_ratio)))
                layer_signs.append(float(sign_ratio.item()))
            else:
                layer_signs.append(0.5)

        benign_idx2 = all_set.copy()
        sign_scores: List[float] = []
        if len(layer_signs) > 0:
            median_sign = float(np.median(layer_signs))
            std_sign = float(np.std(layer_signs))
            for sign in layer_signs:
                score = float(abs((float(sign) - median_sign) / std_sign)) if std_sign > 0 else 0.0
                sign_scores.append(score)
            benign_idx2 = benign_idx2.intersection(set(np.argwhere(np.array(sign_scores) < float(self.config.lasa_lambda_s)).flatten().astype(int).tolist()))

        benign_indices = list(benign_idx1.intersection(benign_idx2))
        if len(benign_indices) == 0:
            benign_indices = list(all_set)
        return benign_indices

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        )  # (K,D)

        # Step 1: gradient clipping (median norm)
        clipped = self._gradient_sanitization_and_clipping(deltas)

        # Step 2: pre-aggregation sparsification on each client's clipped update
        sparse_updates = [_topk_sparsification(u, float(self.config.lasa_sparsity_ratio)) for u in clipped]

        # Step 3: layer-wise detection + mean over benign indices, concatenate
        aggregated_layers: List[Tensor] = []
        benign_fracs: List[float] = []
        benign_counts = torch.zeros(k, dtype=torch.float32)
        for i in range(int(len(self.layer_dims) - 1)):
            start_dim = int(self.layer_dims[i])
            end_dim = int(self.layer_dims[i + 1])
            benign_indices = self._byzantine_detection_layer(sparse_updates, start_dim, end_dim)
            benign_fracs.append(len(benign_indices) / max(1, k))
            if benign_indices:
                benign_counts[torch.tensor(benign_indices, dtype=torch.long)] += 1.0
            benign_layer_params = [clipped[idx][start_dim:end_dim] for idx in benign_indices if idx < k]
            if len(benign_layer_params) > 0:
                aggregated_layer = torch.mean(torch.stack(benign_layer_params, dim=0), dim=0)
            else:
                aggregated_layer = clipped[0][start_dim:end_dim]
            aggregated_layers.append(aggregated_layer)
        delta_agg = torch.cat(aggregated_layers, dim=0)
        # Use per-client benign-layer ratio as buffer-aggregation weights.
        buffer_alpha = benign_counts / (benign_counts.sum() + 1e-12)
        if float(buffer_alpha.sum().item()) <= 0.0:
            buffer_alpha = torch.full((k,), 1.0 / max(1, k), dtype=torch.float32)
        new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
        merged_buffers = _aggregate_nonparam_buffers(global_sd, client_state_dicts, self.param_names, buffer_alpha)
        for kk, vv in merged_buffers.items():
            if kk not in self.param_names:
                new_global_sd[kk] = vv
        self.global_model.load_state_dict(new_global_sd)

        avg_benign_frac = float(np.mean(benign_fracs)) if benign_fracs else 1.0
        num_layers = max(1, int(len(self.layer_dims) - 1))
        d = benign_counts / float(num_layers)  # per-client benign-layer ratio
        m = (d >= 0.5).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        # Report per-client weight using the same normalized benign-layer count
        # used by buffer aggregation; this is more informative than hard-mask uniform.
        alpha = buffer_alpha
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d,
            m=m,
            alpha=alpha,
            phase="lasa | LASA",
            show_detection=True,
            monitor_items=[
                ("Defense", "LASA"),
                ("Avg benign frac/layer", f"{avg_benign_frac:.3f}"),
                ("Kept clients", f"{int(m.sum().item())}/{k}"),
            ],
        )


class FedSECAServer(BaseServer):
    """Efficient FedSECA (CVPR 2025) with fully vectorized operations.

    Key improvements:
    1. Combined sign concordance (ω) + cosine similarity for robust detection
    2. MAD-based outlier detection instead of fixed threshold
    3. Fully vectorized computation - O(1) Python loops
    """

    defense_name = "seca"

    def _compute_trust_scores_vectorized(self, grads: Tensor) -> Tuple[Tensor, Tensor]:
        """Vectorized computation of trust scores for all clients.

        omega[i,j] = (1/D) * Σ sgn(grad_i) * sgn(grad_j)
        cos[i,j] = grad_i · grad_j / (||grad_i|| * ||grad_j||)

        Returns per-client average scores.
        """
        K, D = grads.shape
        grads_f = grads.float()

        # Sign concordance matrix: signs @ signs.T / D
        signs = torch.sign(grads_f)  # (K, D)
        omega_matrix = signs @ signs.T / float(D)  # (K, K)

        # Cosine similarity matrix: (grads @ grads.T) / (norms @ norms.T)
        norms = grads_f.norm(dim=1, keepdim=True)  # (K, 1)
        cos_matrix = grads_f @ grads_f.T  # (K, K)
        cos_matrix = cos_matrix / (norms @ norms.T + 1e-10)  # (K, K)

        # Average over all pairs (including self)
        omega_scores = omega_matrix.mean(dim=1)  # (K,)
        cos_scores = cos_matrix.mean(dim=1)  # (K,)

        return omega_scores, cos_scores

    def _sparsify_vectorized(self, raw_grads: Tensor, clamped_grads: Tensor) -> Tensor:
        """Vectorized sparsification using top-k per client.

        λ_k = γ-quantile of |raw_grads[k]|
        Keeps values where |raw_grads| > λ_k
        """
        K, D = raw_grads.shape
        gamma = float(self.config.fedseca_sparsity_gamma)

        # Per-client thresholds: γ-quantile
        thresholds = torch.quantile(raw_grads.abs(), gamma, dim=1, keepdim=True)  # (K, 1)

        # Keep top (1-γ) fraction
        mask = raw_grads.abs() > thresholds  # (K, D)
        return clamped_grads * mask.float()

    def _robust_detection(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """Detect outliers using MAD (Median Absolute Deviation).

        Returns: (is_benign mask, z_scores)
        """
        median = torch.median(scores)
        mad = torch.median(torch.abs(scores - median))
        mad = max(float(mad), 1e-6)
        z_scores = torch.abs(scores - median) / mad
        is_benign = z_scores < 3.0  # z-score threshold
        return is_benign.float(), z_scores

    def _crise_sign_election(self, grads: Tensor, trust_scores: Tensor) -> Tensor:
        """CRISE: s^j = sgn(Σ_k ρ_k * sgn(g_k^j))"""
        weights = trust_scores.unsqueeze(1).clamp(min=0)  # (K, 1)
        sign_votes = torch.sign(grads)  # (K, D)
        weighted_signs = (sign_votes * weights).sum(dim=0)  # (D,)
        return torch.sign(weighted_signs)

    def _clip_gradients(self, grads: Tensor) -> Tensor:
        norms = grads.norm(dim=1, keepdim=True)  # (K, 1)
        tau = torch.median(norms)
        scale = torch.clamp(tau / (norms + 1e-10), max=1.0)
        return grads * scale

    def _clamp_gradients(self, grads: Tensor) -> Tensor:
        mu = grads.abs().median(dim=0).values  # (D,)
        clamped = grads.abs().clamp(max=mu.unsqueeze(0))
        return torch.sign(grads) * clamped

    def _variance_reduced_sparse(self, grads: Tensor) -> Tensor:
        clipped = self._clip_gradients(grads)
        clamped = self._clamp_gradients(clipped)
        return self._sparsify_vectorized(grads, clamped)

    def _roca(self, sparse_grads: Tensor, elected_signs: Tensor) -> Tensor:
        """RoCA: Robust Coordinate-wise Aggregation.

        From paper:
        - δ_k^j = I(s^j * ġ_k^j > 0)  # alignment indicator
        - g̃^j = Σ_k δ_k^j * ġ_k^j / Σ_k δ_k^j

        When s^j = 0, alignment should be 0 (not 1) to avoid spurious aggregation.
        """
        # alignment indicator: 1 if sign matches, 0 otherwise
        alignment = (elected_signs.unsqueeze(0) * sparse_grads > 0).float()
        # When elected sign is 0, alignment should be 0 to avoid spurious aggregation
        zero_sign_mask = (elected_signs == 0).unsqueeze(0)
        alignment = torch.where(zero_sign_mask, torch.zeros_like(alignment), alignment)

        # Coordinate-wise mean of aligned gradients
        numerator = (alignment * sparse_grads).sum(dim=0)
        denominator = alignment.sum(dim=0)
        aggregated = numerator / (denominator + 1e-10)
        aggregated = torch.where(denominator == 0, torch.zeros_like(aggregated), aggregated)
        return aggregated

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        )  # (K,D)

        # Step 1: Compute trust scores (vectorized O(K²) with matrix ops)
        omega_scores, cos_scores = self._compute_trust_scores_vectorized(deltas)

        # Combined trust score: weighted average of both metrics
        # Cosine similarity is more sensitive to direction differences
        trust_scores = 0.4 * omega_scores + 0.6 * cos_scores

        # Step 2: CRISE sign election using trust scores
        elected_signs = self._crise_sign_election(deltas, trust_scores)

        # Step 3: Robust outlier detection using MAD (Median Absolute Deviation)
        # This adapts to the distribution instead of using fixed threshold
        m_omega, z_omega = self._robust_detection(omega_scores)
        m_cos, z_cos = self._robust_detection(cos_scores)

        # Combined detection: client is benign if BOTH metrics mark it as normal
        m_combined = (m_omega * m_cos)  # 1 if both agree benign, 0 if either flags as outlier

        # If all detected as outliers (adversarial scenario), fall back to trusting majority
        if m_combined.sum() < 1:
            m_combined = torch.ones_like(m_combined)

        # Trust weights proportional to combined trust scores
        trust_weights = trust_scores.clamp(min=0).detach().cpu()
        if trust_weights.sum() > 0:
            trust_weights = trust_weights / trust_weights.sum()
        else:
            trust_weights = torch.ones(k) / k

        # Step 4: Variance Reduction (Clip, Clamp, Sparsify)
        vrs = self._variance_reduced_sparse(deltas)

        # Step 5: RoCA - Robust Coordinate-wise Aggregation
        delta_agg = self._roca(vrs, elected_signs)

        # Buffer aggregation: use trust scores as weights
        buffer_alpha = trust_weights.float()
        new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
        merged_buffers = _aggregate_nonparam_buffers(global_sd, client_state_dicts, self.param_names, buffer_alpha)
        for kk, vv in merged_buffers.items():
            if kk not in self.param_names:
                new_global_sd[kk] = vv
        self.global_model.load_state_dict(new_global_sd)

        # Return detection metrics for monitoring
        d = trust_scores.detach().cpu()
        m = m_combined.detach().cpu()
        alpha = buffer_alpha
        nonzero_ratio = float((delta_agg != 0).float().mean().item())

        # Debug info: compute detection stats
        n_kept = int(m.sum().item())
        n_flagged = k - n_kept
        omega_mean = float(omega_scores.mean().item())
        cos_mean = float(cos_scores.mean().item())

        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d,
            m=m,
            alpha=alpha,
            phase="seca | FedSECA",
            show_detection=True,
            monitor_items=[
                ("Defense", "FedSECA (Fixed)"),
                ("Nonzero ratio", f"{nonzero_ratio:.3f}"),
                ("Kept/Flagged", f"{n_kept}/{n_flagged}"),
                ("Omega(avg)", f"{omega_mean:.3f}"),
                ("CosSim(avg)", f"{cos_mean:.3f}"),
            ],
        )


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


class FLDefenderServer(BaseServer):
    """Port of FL-Defender from experiment/FL-Byzantine-Library/aggregators/fl_defender.py.

    Keeps the same steps but avoids sklearn dependency (standardization + PCA via SVD).
    """

    defense_name = "fld"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self.n_clients = int(config.num_clients)
        self.score_history = np.zeros(self.n_clients, dtype=np.float64)
        self.rounds = 0

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        )  # (K,D)

        grads_np = deltas.detach().cpu().numpy()
        valid_mask = np.all(np.isfinite(grads_np), axis=1)
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size == 0:
                delta_agg = torch.mean(deltas, dim=0)
                new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
                self.global_model.load_state_dict(new_global_sd)
                m = torch.ones(k)
                alpha = torch.full((k,), 1.0 / max(1, k))
                return RoundStats(
                    center_norm=float("nan"),
                    z_var=0.0,
                    ae_loss=float("nan"),
                    svdd_loss=float("nan"),
                    d=torch.zeros(k),
                    m=m,
                    alpha=alpha,
                    phase="fld | FL-Defender (fallback)",
                    show_detection=False,
                    monitor_items=[("Defense", "FL-Defender"), ("Invalid grads", f"{n_invalid}/{k}")],
                )
            grads_np_valid = grads_np[valid_indices]
        else:
            valid_indices = np.arange(k)
            grads_np_valid = grads_np

        n = grads_np_valid.shape[0]
        cs = _cosine_similarity_matrix(grads_np_valid) - np.eye(n)
        cs_scaled = _standardize_matrix(cs)
        cs_pca = _pca_2d_via_svd(cs_scaled, n_components=int(self.config.fldefender_pca_components))
        centroid = np.median(cs_pca, axis=0, keepdims=True)
        scores = _cosine_similarity_matrix(np.vstack([centroid, cs_pca]))[0, 1:]

        # accumulate reputation
        if n == self.n_clients and n_invalid == 0:
            self.score_history += scores
        else:
            self.score_history = np.zeros(self.n_clients, dtype=np.float64)
            self.score_history[valid_indices] = scores

        q1 = float(np.quantile(self.score_history, float(self.config.fldefender_q1)))
        trust = self.score_history - q1
        max_trust = float(trust.max())
        if max_trust > 0:
            trust = trust / max_trust
        trust = np.clip(trust, 0.0, None)
        trust_weights = trust[valid_indices]

        total_weight = float(trust_weights.sum())
        if total_weight > 0:
            w = torch.tensor(trust_weights, dtype=deltas.dtype)
            w = w / (w.sum() + 1e-12)
            delta_agg = (w.unsqueeze(1) * deltas[valid_indices]).sum(dim=0)
        else:
            delta_agg = torch.mean(deltas[valid_indices], dim=0)
        buffer_alpha = torch.zeros(k, dtype=torch.float32)
        if total_weight > 0:
            buffer_alpha[valid_indices] = torch.tensor(trust_weights, dtype=torch.float32)
        else:
            buffer_alpha[valid_indices] = 1.0
        new_global_sd = _apply_flat_param_delta_to_global(global_sd, delta_agg, self.param_names)
        merged_buffers = _aggregate_nonparam_buffers(global_sd, client_state_dicts, self.param_names, buffer_alpha)
        for kk, vv in merged_buffers.items():
            if kk not in self.param_names:
                new_global_sd[kk] = vv
        self.global_model.load_state_dict(new_global_sd)

        active_ratio = float(np.mean(trust_weights > 0)) if trust_weights.size > 0 else 0.0
        mean_trust = float(np.mean(trust_weights)) if trust_weights.size > 0 else 0.0
        full_trust = np.zeros(k, dtype=np.float64)
        full_trust[valid_indices] = trust_weights
        d = torch.tensor(full_trust, dtype=torch.float32)
        m = (d > 0).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = d.clone()
        if float(alpha.sum().item()) > 0.0:
            alpha = alpha / alpha.sum()
        else:
            alpha = m / (m.sum() + 1e-12)
        self.rounds += 1
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d,
            m=m,
            alpha=alpha,
            phase="fld | FL-Defender",
            show_detection=True,
            monitor_items=[
                ("Defense", "FL-Defender"),
                ("Active ratio", f"{active_ratio:.3f}"),
                ("Mean trust", f"{mean_trust:.3f}"),
                ("Invalid grads", f"{n_invalid}/{k}"),
            ],
        )


class FedAvgServer(BaseServer):
    defense_name = "avg"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd, m, alpha = aggregate_updates_with_info(client_state_dicts, method="fedavg")
        self.global_model.load_state_dict(global_sd)
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=torch.zeros(k),
            m=m,
            alpha=alpha,
            phase="avg | FedAvg",
            show_detection=True,
            monitor_items=[
                ("Defense", "FedAvg"),
                ("Clients Kept", f"{int(m.sum().item())}/{k}"),
                ("Uniform Weight", f"{(1.0 / max(1, k)):.6f}"),
            ],
        )


class TrimmedMeanServer(BaseServer):
    defense_name = "tm"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        num_byzantine = (
            self.config.trimmed_mean_num_byzantine
            if self.config.trimmed_mean_num_byzantine is not None
            else max(0, self.config.num_clients - self.config.num_benign)
        )
        global_sd, m, alpha = aggregate_updates_with_info(
            client_state_dicts,
            method="trimmed_mean",
            trim_ratio=self.config.trimmed_mean_ratio,
            num_byzantine=num_byzantine,
        )
        self.global_model.load_state_dict(global_sd)
        kept_per_coordinate = k - 2 * int(num_byzantine)
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=torch.zeros(k),
            m=m,
            alpha=alpha,
            phase="tm | Trimmed mean",
            show_detection=True,
            monitor_items=[
                ("Defense", "Trimmed Mean"),
                ("Byzantine b", str(int(num_byzantine))),
                ("Kept/coord", f"{kept_per_coordinate}/{k}"),
                ("Clients Kept", f"{int(m.sum().item())}/{k}"),
            ],
        )


class MultiKrumServer(BaseServer):
    defense_name = "mk"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        num_byzantine = (
            self.config.krum_num_byzantine
            if self.config.krum_num_byzantine is not None
            else max(0, self.config.num_clients - self.config.num_benign)
        )
        krum_neighbors = k - num_byzantine - 2
        krum_scores = compute_multi_krum_scores(client_state_dicts, num_byzantine=num_byzantine).detach().cpu()
        global_sd, m, alpha = aggregate_updates_with_info(
            client_state_dicts,
            method="multi_krum",
            num_byzantine=num_byzantine,
            num_selected=self.config.multi_krum_num_selected,
        )
        self.global_model.load_state_dict(global_sd)
        selected_count = int(m.sum().item())
        selected_m = selected_count
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=krum_scores,
            m=m,
            alpha=alpha,
            phase="mk | Multi-Krum",
            show_detection=True,
            monitor_items=[
                ("Defense", "Multi-Krum"),
                ("Byzantine f", str(int(num_byzantine))),
                ("Score Neighbors", f"n-f-2 = {int(krum_neighbors)}"),
                ("Selected m", str(int(selected_m))),
                ("Clients Kept", f"{selected_count}/{k}"),
            ],
        )


class SVDDServer(BaseServer):
    """Two-phase AE-SVDD defense server."""

    defense_name = "svdd"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
        svdd_feature_extractor: Optional[Callable[[Dict[str, Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._svdd_feat: Callable[[Dict[str, Tensor]], Tensor] = (
            svdd_feature_extractor or extract_bn_features
        )
        # 限制潜在空间维度，避免高维距离退化
        latent_dim = min(config.latent_dim, 64)
        self.ae = AutoEncoder(d_bn=d_bn, latent_dim=latent_dim).to(self.device)

        self.c: Optional[Tensor] = None

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )

    def _state_dict_for_clients(self) -> Dict[str, Tensor]:
        """Return a detached CPU copy of global state_dict for broadcasting."""

        sd = self.global_model.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    def phase1_step(
        self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, Tensor, Tensor]:
        """AE warm-up: train AE and FedAvg using only the closest clients in feature space.

        Distance = L2 norm to the coordinate-wise median of robust-scaled SVDD features
        across all clients this round. Keep ``ae_warmup_keep_ratio`` of clients with
        smallest distance (at least one).

        Returns:
            center_norm, z_variance, ae_loss,
            per-client reconstruction loss (all K, for monitoring), keep_mask (bool K)
        """

        X = build_svdd_feature_matrix(client_state_dicts, self._svdd_feat)  # (K, D_feat)
        X = robust_scale_features(X)
        K = int(X.shape[0])
        ratio = float(getattr(self.config, "ae_warmup_keep_ratio", 0.8))
        ratio = min(max(ratio, 1e-6), 1.0)
        num_keep = max(1, min(K, int(round(ratio * K))))

        ref = X.median(dim=0).values
        distances = torch.norm(X - ref.unsqueeze(0), dim=1)
        _, idx_keep = torch.topk(distances, k=num_keep, largest=False)
        idx_keep = torch.sort(idx_keep).values

        keep_mask = torch.zeros(K, dtype=torch.bool)
        keep_mask[idx_keep] = True

        self.ae.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            x_hat_cur = self.ae(X_dev)
            per_client_loss = (x_hat_cur - X_dev).abs().sum(dim=1)  # (K,)

        X_train = X[idx_keep].to(self.device)

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = (x_hat - X_train).abs().sum(dim=1)
        loss = per_sample_loss.mean()

        self.optimizer_ae.zero_grad()
        loss.backward()
        clip_grad_norm_(self.ae.parameters(), self.config.ae_grad_clip)
        self.optimizer_ae.step()

        selected_sds = [client_state_dicts[int(i)] for i in idx_keep.tolist()]
        alpha_sel = torch.full((num_keep,), 1.0 / float(num_keep))
        global_sd = weighted_fedavg(selected_sds, alpha_sel)
        self.global_model.load_state_dict(global_sd)

        with torch.no_grad():
            self.ae.eval()
            Z = self.ae.encode(X.to(self.device))
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return center_norm, z_var, float(loss.item()), per_client_loss.detach().cpu(), keep_mask.detach().cpu()
    
    def init_center(self, client_state_dicts: List[Dict[str, Tensor]]) -> Tuple[float, float]:
        """Initialize SVDD center c using well-reconstructed clients."""

        X = build_svdd_feature_matrix(client_state_dicts, self._svdd_feat)
        X = robust_scale_features(X).to(self.device)
        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X)
            x_hat = self.ae(X)
            recon_error = (x_hat - X).abs().sum(dim=1)
            med = torch.median(recon_error)
            init_mask = recon_error <= med
            c = Z[init_mask].mean(dim=0)
            c[c.abs() < 0.01] = 0.01

        self.c = c.detach()
        center_norm = float(self.c.norm().item())
        z_var = float(Z.var().item())
        return center_norm, z_var

    def phase2_step(
        self, svdd_round: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, float, Tensor, Tensor, Tensor]:
        """Run one SVDD-filtered aggregation round.

        Returns:
            center_norm, z_variance, svdd_loss_value, recon_loss_value,
            d, M, alpha
        """

        assert self.c is not None, "SVDD center c must be initialized before Phase 2."

        X = build_svdd_feature_matrix(client_state_dicts, self._svdd_feat)
        X = robust_scale_features(X)

        # Embeddings without grad
        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X.to(self.device))
        c = self.c.to(self.device)
        d = ((Z - c) ** 2).sum(dim=1)  # (K,)
        med_d = torch.median(d)
        mad_d = 1.4826 * torch.median((d - med_d).abs())
        mad_d = torch.clamp(mad_d, min=1e-6)
        p_tau = min(1.0, svdd_round / float(self.config.svdd_warmup_rounds))
        if self.config.tau_start > 0.0 and self.config.tau_end > 0.0:
            tau = self.config.tau_start - p_tau * (self.config.tau_start - self.config.tau_end)
        else:
            tau = self.config.tau_multiplier
        threshold = med_d + tau * mad_d

        M = (d <= threshold).float()
        if M.sum() < 1:
            M = torch.ones_like(M)

        alpha = M / (M.sum() + 1e-12)

        trusted = M > 0.5
        if trusted.sum() > 0:
            with torch.no_grad():
                c_new = Z[trusted].mean(dim=0)
                c_updated = self.config.center_ema_decay * c + (1.0 - self.config.center_ema_decay) * c_new
                c_updated[c_updated.abs() < 0.01] = 0.01
                self.c = c_updated.detach()

        self.ae.train()
        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = False

        trusted_cpu = trusted.detach().cpu()
        X_trusted = X[trusted_cpu]
        Z_trusted = self.ae.encode(X_trusted.to(self.device))
        svdd_loss = ((Z_trusted - self.c.detach().to(self.device)) ** 2).sum(dim=1).mean()

        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = True

        X_cur = X.to(self.device)
        X_cur_hat = self.ae(X_cur)
        recon_per_sample = (X_cur_hat - X_cur).abs().mean(dim=1)
        q80 = torch.quantile(recon_per_sample.detach(), 0.8)
        keep = recon_per_sample <= q80
        recon_loss = recon_per_sample[keep].mean()

        total_loss = svdd_loss + self.config.svdd_recon_lambda * recon_loss

        self.optimizer_ae.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.ae.parameters(), self.config.svdd_grad_clip)
        self.optimizer_ae.step()

        # Weighted aggregation
        global_sd = weighted_fedavg(client_state_dicts, alpha.detach().cpu())
        self.global_model.load_state_dict(global_sd)

        center_norm = float(self.c.norm().item())
        z_var = float(Z.var().item())

        return (
            center_norm,
            z_var,
            float(svdd_loss.item()),
            float(recon_loss.item()),
            d.detach().cpu(),
            M.detach().cpu(),
            alpha.detach().cpu(),
        )

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        phase1_rounds = self.config.phase1_rounds
        if round_idx <= phase1_rounds:
            center_norm, z_var, ae_loss, d, keep_mask = self.phase1_step(round_idx, client_state_dicts)
            kmf = keep_mask.float()
            alpha = kmf / (kmf.sum() + 1e-12)
            n_kept = int(kmf.sum().item())
            k_tot = len(client_state_dicts)
            return RoundStats(
                center_norm=center_norm,
                z_var=z_var,
                ae_loss=ae_loss,
                svdd_loss=float("nan"),
                d=d,
                m=keep_mask.float(),
                alpha=alpha,
                phase="svdd | AE Warm-up",
                show_detection=True,
                monitor_items=[
                    ("Defense", "SVDD"),
                    ("AE+FedAvg clients", f"{n_kept}/{k_tot}"),
                    ("Center L2-Norm", f"{center_norm:.6f}"),
                    ("Z-Space Variance", f"{z_var:.6f}"),
                    ("AE L1-Loss", f"{ae_loss:.6f}"),
                ],
            )

        # Phase 2: ensure center is initialized, then immediately run SVDD filtering.
        if self.c is None:
            # Lazily initialize center using the first post-warmup batch of updates.
            center_norm, z_var = self.init_center(client_state_dicts)
            svdd_round = 1
        else:
            svdd_round = round_idx - phase1_rounds
        center_norm, z_var, svdd_loss, _recon_loss, d, m, alpha = self.phase2_step(
            svdd_round, client_state_dicts
        )
        k_tot = len(client_state_dicts)
        kept = int(m.sum().item())
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=float("nan"),
            svdd_loss=svdd_loss,
            d=d,
            m=m,
            alpha=alpha,
            phase="svdd | Filtering",
            show_detection=True,
            monitor_items=[
                ("Defense", "SVDD (hard)"),
                ("Kept clients", f"{kept}/{k_tot}"),
                ("Center L2-Norm", f"{center_norm:.6f}"),
                ("Z-Space Variance", f"{z_var:.6f}"),
                ("SVDD Loss", f"{svdd_loss:.6f}"),
            ],
        )


DEFENSE_REGISTRY: Dict[str, Type[BaseServer]] = {
    "avg": FedAvgServer,
    "tm": TrimmedMeanServer,
    "mk": MultiKrumServer,
    "svdd": SVDDServer,
    "lasa": LASAServer,
    "seca": FedSECAServer,
    "fld": FLDefenderServer,
}

# Backward compatibility
FederatedServer = SVDDServer

