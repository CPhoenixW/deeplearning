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
    from .fixed_descriptor import FixedHierarchicalMultiViewDescriptor
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
    from fixed_descriptor import FixedHierarchicalMultiViewDescriptor
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

    @property
    def aggregation_device(self) -> torch.device:
        if self.config.cuda_aggregation and self.device.type == "cuda":
            return self.device
        return torch.device("cpu")

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


class AlignInsServer(BaseServer):
    """AlignIns-style update filtering (TDA + MPSA) on flattened floating deltas.

    Ported conservatively from experiment/AlignIns/src/aggregation.py:
    - TDA: cosine alignment of each client delta with the mean update direction.
    - MPSA: sign agreement with the majority sign on top-|Δ| coordinates.
    Outliers are flagged via MAD-z thresholds; kept clients are norm-clipped and averaged.
    """

    defense_name = "alignins"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_delta(global_sd, sd) for sd in client_state_dicts], dim=0
        )  # (K, D)

        agg_dir = deltas.mean(dim=0)
        agg_norm = agg_dir.norm().clamp(min=1e-12)
        delta_norms = deltas.norm(dim=1).clamp(min=1e-12)
        tda = (deltas @ agg_dir) / (delta_norms * agg_norm)

        major_sign = torch.sign(deltas.sum(dim=0))
        sparsity = float(self.config.alignins_sparsity)
        sparsity = min(max(sparsity, 1e-6), 1.0)
        k_dim = max(1, int(sparsity * int(deltas.shape[1])))
        k_dim = min(k_dim, int(deltas.shape[1]))

        mpsa = torch.empty(k, dtype=torch.float32)
        for i in range(k):
            _, idx = torch.topk(deltas[i].abs(), k_dim)
            agree = (torch.sign(deltas[i, idx]) == major_sign[idx]).float().mean()
            mpsa[i] = agree

        mz_mpsa = _mad_zscores(mpsa)
        mz_tda = _mad_zscores(tda.float())
        lambda_s = float(self.config.alignins_lambda_s)
        lambda_c = float(self.config.alignins_lambda_c)
        m = ((mz_mpsa < lambda_s) & (mz_tda < lambda_c)).float()
        if m.sum() < 1:
            m = torch.ones_like(m)

        d = torch.max(mz_mpsa, mz_tda)
        alpha = m / (m.sum() + 1e-12)

        kept_idx = (m > 0.5).nonzero(as_tuple=False).view(-1)
        kept_updates = deltas[kept_idx]
        updates_norm = kept_updates.norm(dim=1, keepdim=True)
        norm_clip = updates_norm.median(dim=0).values.clamp(min=1e-12)
        clipped_norm = torch.clamp(updates_norm, max=norm_clip)
        kept_updates = kept_updates / updates_norm * clipped_norm
        delta_agg = kept_updates.mean(dim=0)

        new_global_sd = _apply_flat_delta_to_global(global_sd, delta_agg)
        self.global_model.load_state_dict(new_global_sd)

        n_kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase="alignins | AlignIns",
            show_detection=True,
            monitor_items=[
                ("Defense", "AlignIns"),
                ("TDA (avg cos)", f"{float(tda.mean().item()):.4f}"),
                ("MPSA (avg)", f"{float(mpsa.mean().item()):.4f}"),
                ("Kept clients", f"{n_kept}/{k}"),
            ],
        )


class BNGuardServer(BaseServer):
    """Lightweight BN-statistics OOD/backdoor detector (median + MAD distance)."""

    defense_name = "bnguard"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        X = build_svdd_feature_matrix(client_state_dicts, extract_bn_features)
        X = robust_scale_features(X)

        ref = X.median(dim=0).values
        d = torch.norm(X - ref.unsqueeze(0), dim=1)
        med_d = torch.median(d)
        mad_d = 1.4826 * torch.median((d - med_d).abs())
        mad_d = torch.clamp(mad_d, min=1e-6)
        tau = float(self.config.bnguard_tau)
        threshold = med_d + tau * mad_d

        m = (d <= threshold).float()
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = m / (m.sum() + 1e-12)

        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        n_kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase="bnguard | BNGuard",
            show_detection=True,
            monitor_items=[
                ("Defense", "BNGuard"),
                ("Dist threshold", f"{float(threshold.item()):.4f}"),
                ("Kept clients", f"{n_kept}/{k}"),
            ],
        )


class FLGMMServer(BaseServer):
    """FLGMM-style GMM/SPC filtering over model-distance statistics."""

    defense_name = "flgmm"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._distance_history: List[List[float]] = []
        self._ucl: Optional[float] = None

    def _ensure_history(self, k: int) -> None:
        if len(self._distance_history) != k:
            self._distance_history = [[] for _ in range(k)]
            self._ucl = None

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        self._ensure_history(k)

        # FLGMM first compares local models against the all-client temporary
        # FedAvg model, then models this 1-D distance distribution.
        temp_global_sd = weighted_fedavg(
            client_state_dicts, torch.full((k,), 1.0 / float(k), dtype=torch.float32)
        )
        raw_dist = np.array(
            [_state_l2_distance(sd, temp_global_sd) for sd in client_state_dicts],
            dtype=np.float64,
        )
        normal_mask, normal_mean, normal_std = _fit_1d_gmm_largest_cluster(
            raw_dist, n_iter=int(self.config.flgmm_em_iters)
        )
        z_dist = (raw_dist - normal_mean) / max(normal_std, 1e-6)
        for i, z in enumerate(z_dist.tolist()):
            self._distance_history[i].append(float(z))

        warmup = max(1, int(self.config.flgmm_warmup_rounds))
        if round_idx <= warmup:
            m_np = normal_mask.astype(np.float32)
            phase = "flgmm | GMM Warm-up"
            if round_idx == warmup:
                all_z = np.array(
                    [z for hist in self._distance_history for z in hist],
                    dtype=np.float64,
                )
                normal_z_mask, _, _ = _fit_1d_gmm_largest_cluster(
                    all_z, n_iter=int(self.config.flgmm_em_iters)
                )
                normal_z = all_z[normal_z_mask]
                if normal_z.size == 0:
                    normal_z = all_z
                self._ucl = float(
                    np.mean(normal_z)
                    + float(self.config.flgmm_control_l) * max(float(np.std(normal_z)), 1e-6)
                )
        else:
            ucl = float(self._ucl) if self._ucl is not None else float(
                np.mean(z_dist) + float(self.config.flgmm_control_l) * max(float(np.std(z_dist)), 1e-6)
            )
            m_np = (z_dist < ucl).astype(np.float32)
            phase = "flgmm | SPC Filtering"

        m = torch.tensor(m_np, dtype=torch.float32)
        if m.sum() < 1:
            m = torch.ones_like(m)
        alpha = m / (m.sum() + 1e-12)
        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        d = torch.tensor(z_dist, dtype=torch.float32)
        kept = int(m.sum().item())
        ucl_label = "pending" if self._ucl is None else f"{self._ucl:.4f}"
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase=phase,
            show_detection=True,
            monitor_items=[
                ("Defense", "FLGMM"),
                ("GMM normal mean", f"{normal_mean:.4f}"),
                ("GMM normal std", f"{normal_std:.4f}"),
                ("SPC UCL", ucl_label),
                ("Kept clients", f"{kept}/{k}"),
            ],
        )


class FLANDERSServer(BaseServer):
    """FLANDERS-style MAR forecast filtering over client parameter time series."""

    defense_name = "flanders"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._history: List[List[np.ndarray]] = []
        self._sample_idx: Optional[Tensor] = None

    def _ensure_state(self, k: int, flat_dim: int) -> None:
        if len(self._history) != k:
            self._history = [[] for _ in range(k)]
        if self._sample_idx is None:
            sample_n = int(self.config.flanders_sampling)
            if sample_n <= 0 or sample_n >= flat_dim:
                self._sample_idx = None
            else:
                gen = torch.Generator()
                gen.manual_seed(int(self.config.seed))
                self._sample_idx = torch.randperm(flat_dim, generator=gen)[:sample_n]

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd = self.state_dict_for_clients()
        first_flat = _flatten_delta(global_sd, client_state_dicts[0])
        self._ensure_state(k, int(first_flat.numel()))

        sampled_updates: List[Tensor] = [first_flat if self._sample_idx is None else first_flat[self._sample_idx]]
        for sd in client_state_dicts[1:]:
            sampled_updates.append(_sample_flat_delta(global_sd, sd, self._sample_idx))

        current = np.stack([u.detach().cpu().numpy().astype(np.float64) for u in sampled_updates], axis=0)
        for i, row in enumerate(current):
            self._history[i].append(row)
            window = int(self.config.flanders_window)
            if window > 0 and len(self._history[i]) > window:
                self._history[i] = self._history[i][-window:]

        min_hist = min(len(h) for h in self._history) if self._history else 0
        if min_hist < 2:
            scores = np.zeros(k, dtype=np.float64)
            m = torch.ones(k, dtype=torch.float32)
            phase = "flanders | Warm-up"
        else:
            hist = np.stack([np.stack(h[-min_hist:], axis=0) for h in self._history], axis=0)
            params_tensor = np.transpose(hist, (0, 2, 1))  # (clients, params, time)
            ground_truth = params_tensor[:, :, -1].copy()
            predicted = _mar_forecast(
                params_tensor[:, :, :-1],
                pred_step=1,
                alpha=float(self.config.flanders_alpha),
                beta=float(self.config.flanders_beta),
                maxiter=int(self.config.flanders_maxiter),
            )[:, :, 0]
            scores = np.sqrt(np.sum((ground_truth - predicted) ** 2, axis=1))
            keep_n_cfg = self.config.flanders_num_clients_to_keep
            keep_n = int(keep_n_cfg) if keep_n_cfg is not None else int(self.config.num_benign)
            keep_n = max(1, min(k, keep_n))
            keep_idx = np.argsort(scores)[:keep_n]
            m_np = np.zeros(k, dtype=np.float32)
            m_np[keep_idx] = 1.0
            m = torch.tensor(m_np, dtype=torch.float32)
            phase = "flanders | MAR Filtering"

        alpha = m / (m.sum() + 1e-12)
        global_sd_new = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd_new)

        d = torch.tensor(scores, dtype=torch.float32)
        kept = int(m.sum().item())
        return RoundStats(
            center_norm=float("nan"),
            z_var=0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=d.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase=phase,
            show_detection=True,
            monitor_items=[
                ("Defense", "FLANDERS"),
                ("MAR window used", f"{min_hist}"),
                ("Sampled params", f"{int(current.shape[1])}"),
                ("MAR score avg", f"{float(np.mean(scores)):.4f}"),
                ("Kept clients", f"{kept}/{k}"),
            ],
        )


class FedDMCServer(BaseServer):
    """Data-free multi-view malicious-client detection (FedDMC-style).

    FedDMC (Mu et al., IEEE TDSC) motivates combining client behaviour
    statistics, anomaly detection and dynamic trust rather than designing a
    detector for one backdoor signature.  This implementation keeps the
    server data-free and dependency-free:

    * **magnitude**: robust z-score of the update norm;
    * **direction**: disagreement with the coordinate-wise median update;
    * **sign**: disagreement with the majority sign on update coordinates;
    * **sparsity**: abnormal fraction of changed coordinates;
    * **temporal behaviour**: deviation from each client's EMA descriptor.

    The views are fused into a non-negative anomaly score.  A median/MAD
    threshold creates a hard isolation mask after a short warm-up, while
    ``1 / (1 + score)`` gives the remaining clients dynamic aggregation
    weights.  It therefore handles noise, sign/label/model poisoning, LIE and
    backdoor uploads without a root/validation dataset or attack labels.
    """

    defense_name = "dmc"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._raw_ema: Optional[Tensor] = None
        self._score_ema: Optional[Tensor] = None

    @staticmethod
    def _robust_z(values: Tensor) -> Tensor:
        """Absolute median/MAD z-score, finite and stable for constant rows."""
        values = torch.nan_to_num(values.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        med = values.median()
        mad = (values - med).abs().median()
        # 1.4826 makes MAD comparable to std for a Gaussian; the floor also
        # prevents a single outlier from producing NaN/Inf on small batches.
        scale = torch.clamp(1.4826 * mad, min=1e-6)
        return ((values - med).abs() / scale).clamp(max=1e6)

    def _views(self, deltas: Tensor) -> Tensor:
        """Return (clients, 5) robust anomaly views from (clients, params)."""
        k, d = deltas.shape
        finite = torch.isfinite(deltas).all(dim=1)
        safe = torch.nan_to_num(deltas.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # Coordinate-wise median is translation-free and avoids O(K^2) pair
        # matrices, which matters for large models.
        center = safe.median(dim=0).values
        center_norm = center.norm().clamp_min(1e-12)
        norms = safe.norm(dim=1)
        norm_view = self._robust_z(norms)

        dot = safe @ center
        cos = dot / (safe.norm(dim=1).clamp_min(1e-12) * center_norm)
        direction_view = self._robust_z(1.0 - cos.clamp(-1.0, 1.0))

        # Ignore coordinates where the median sign is zero; this avoids
        # rewarding all-zero coordinates in sparse or frozen layers.
        sign_center = torch.sign(safe.sum(dim=0))
        active = sign_center != 0
        if bool(active.any()):
            signs = torch.sign(safe[:, active])
            sign_agreement = (signs * sign_center[active]).mean(dim=1)
            sign_view = self._robust_z(1.0 - sign_agreement)
        else:
            sign_view = torch.zeros(k, dtype=torch.float32)

        nonzero = (safe.abs() > 1e-12).float().mean(dim=1)
        sparsity_view = self._robust_z(nonzero)

        raw_sign = 1.0 - sign_agreement if bool(active.any()) else torch.zeros(k)
        raw = torch.stack([norms, 1.0 - cos, raw_sign, nonzero], dim=1)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0)
        if self._raw_ema is None or self._raw_ema.shape != raw.shape:
            temporal_view = torch.zeros(k, dtype=torch.float32)
            self._raw_ema = raw.detach().clone()
        else:
            temporal_raw = (raw - self._raw_ema).abs().mean(dim=1)
            temporal_view = self._robust_z(temporal_raw)
            decay = float(np.clip(getattr(self.config, "dmc_ema_decay", 0.8), 0.0, 1.0))
            self._raw_ema = decay * self._raw_ema + (1.0 - decay) * raw.detach()

        views = torch.stack(
            [norm_view, direction_view, sign_view, sparsity_view, temporal_view], dim=1
        )
        views[~finite] = 1e6
        return torch.nan_to_num(views, nan=0.0, posinf=1e6, neginf=0.0)

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        if k == 0:
            raise ValueError("FedDMC requires at least one client update")
        global_sd = self.state_dict_for_clients()
        deltas = torch.stack(
            [_flatten_param_delta(global_sd, sd, self.param_names) for sd in client_state_dicts], dim=0
        ).float()
        valid_rows = torch.isfinite(deltas).all(dim=1)
        views = self._views(deltas)

        weights = torch.tensor(
            [
                float(getattr(self.config, "dmc_norm_weight", 1.0)),
                float(getattr(self.config, "dmc_direction_weight", 1.0)),
                float(getattr(self.config, "dmc_sign_weight", 1.0)),
                float(getattr(self.config, "dmc_sparsity_weight", 0.5)),
                float(getattr(self.config, "dmc_temporal_weight", 1.0)),
            ], dtype=torch.float32
        ).clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            weights.fill_(1.0)
        weights = weights / weights.sum()
        score = (views * weights.view(1, -1)).sum(dim=1)

        score_decay = float(np.clip(getattr(self.config, "dmc_score_ema_decay", 0.7), 0.0, 1.0))
        if self._score_ema is None or self._score_ema.numel() != k:
            self._score_ema = score.detach().clone()
        else:
            self._score_ema = score_decay * self._score_ema + (1.0 - score_decay) * score.detach()
        score = self._score_ema.clone()

        warmup = max(0, int(getattr(self.config, "dmc_warmup_rounds", 3)))
        tau = max(0.0, float(getattr(self.config, "dmc_tau", 3.0)))
        med = score.median()
        mad = (score - med).abs().median()
        threshold = med + tau * max(float(1.4826 * mad.item()), 1e-6)
        m = torch.ones(k, dtype=torch.float32)
        if round_idx > warmup and float(mad.item()) > 1e-8:
            m = (score <= threshold).float()
        # Non-finite uploads are never trusted, even when the robust spread is
        # degenerate (for example, a two-client round with one NaN row).
        m[~valid_rows] = 0.0
        # Never let a degenerate batch discard every participant; keep the
        # lowest-scoring clients as a deterministic safety fallback.
        min_keep = max(1, min(k, int(getattr(self.config, "dmc_min_keep", 1))))
        if int(m.sum().item()) < min_keep:
            m.zero_()
            valid_idx = torch.where(valid_rows)[0]
            if valid_idx.numel() > 0:
                keep_idx = valid_idx[torch.argsort(score[valid_idx])[:min_keep]]
            else:
                # If every upload is invalid, weighted_fedavg still receives a
                # finite fallback after _views() sanitization; mark the first
                # slot to keep the mask/alpha contract well-defined.
                keep_idx = torch.argsort(score)[:min_keep]
            m[keep_idx] = 1.0

        trust = 1.0 / (1.0 + score.clamp_min(0.0))
        trust = trust * m
        if float(trust.sum().item()) <= 1e-12:
            trust = m.clone()
        alpha = trust / trust.sum().clamp_min(1e-12)

        # Avoid ``0 * NaN`` poisoning the weighted average.  Invalid uploads
        # are replaced by the pre-round global snapshot; their mask is zero
        # whenever at least one valid client exists.
        aggregation_sds: List[Dict[str, Tensor]] = []
        for row_valid, sd in zip(valid_rows.tolist(), client_state_dicts):
            if row_valid:
                aggregation_sds.append(sd)
                continue
            cleaned: Dict[str, Tensor] = {}
            for name, g in global_sd.items():
                value = sd[name].detach().cpu()
                if value.is_floating_point():
                    cleaned[name] = torch.where(
                        torch.isfinite(value), value, g.detach().cpu()
                    ).clone()
                else:
                    cleaned[name] = g.detach().cpu().clone()
            aggregation_sds.append(cleaned)
        global_sd_new = weighted_fedavg(
            aggregation_sds, alpha.detach().cpu(), device=self.aggregation_device
        )
        self.global_model.load_state_dict(global_sd_new)
        kept = int(m.sum().item())
        phase = "dmc | Warm-up" if round_idx <= warmup else "dmc | Multi-view Filtering"
        return RoundStats(
            center_norm=float("nan"),
            z_var=float(torch.var(score).item()) if k > 1 else 0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=score.detach().cpu(),
            m=m.detach().cpu(),
            alpha=alpha.detach().cpu(),
            phase=phase,
            show_detection=True,
            monitor_items=[
                ("Defense", "FedDMC (multi-view)"),
                ("Score threshold", f"{threshold:.4f}"),
                ("Kept clients", f"{kept}/{k}"),
                ("Views", "norm+direction+sign+sparsity+temporal"),
                ("Score avg", f"{float(score.mean().item()):.4f}"),
            ],
        )


class FedAvgServer(BaseServer):
    defense_name = "avg"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd, m, alpha = aggregate_updates_with_info(
            client_state_dicts,
            method="fedavg",
            device=self.aggregation_device,
        )
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
            device=self.aggregation_device,
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
        krum_scores_dev = compute_multi_krum_scores(
            client_state_dicts,
            num_byzantine=num_byzantine,
            device=self.aggregation_device,
        )
        krum_scores = krum_scores_dev.detach().cpu()
        selected_m = (
            self.config.multi_krum_num_selected
            if self.config.multi_krum_num_selected is not None
            else krum_neighbors
        )
        selected_m = max(1, min(int(selected_m), k))
        selected = torch.topk(krum_scores_dev, k=selected_m, largest=False).indices.detach().cpu()
        m = torch.zeros(k)
        m[selected] = 1.0
        alpha = m / m.sum()
        selected_sds = [client_state_dicts[int(i)] for i in selected.tolist()]
        global_sd = weighted_fedavg(
            selected_sds,
            torch.full((selected_m,), 1.0 / float(selected_m)),
            device=self.aggregation_device,
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
        self._fixed_descriptor: Optional[FixedHierarchicalMultiViewDescriptor] = None
        feature_mode = str(getattr(config, "svdd_feature_mode", "task")).lower().strip()
        if feature_mode == "fixed_projection":
            descriptor_device_name = str(
                getattr(config, "param_descriptor_device", "cpu")
            ).lower().strip()
            if descriptor_device_name == "auto":
                descriptor_device = self.device
            elif descriptor_device_name in {"cpu", "cuda"}:
                descriptor_device = torch.device(descriptor_device_name)
            else:
                raise ValueError(
                    "param_descriptor_device must be 'cpu', 'cuda', or 'auto'."
                )
            if descriptor_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA parameter descriptor requested but CUDA is unavailable.")
            self._fixed_descriptor = FixedHierarchicalMultiViewDescriptor(
                self.state_dict_for_clients(),
                parameter_names=self.param_names,
                output_dim=int(config.param_descriptor_dim),
                seed=int(config.param_descriptor_seed),
                projection_device=descriptor_device,
            )
        # 限制潜在空间维度，避免高维距离退化
        latent_dim = min(config.latent_dim, 64)
        self.ae = AutoEncoder(d_bn=d_bn, latent_dim=latent_dim).to(self.device)

        self.c: Optional[Tensor] = None

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )

    def _build_input_matrix(
        self, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tensor:
        """Build absolute-model or pre-round model-delta features."""

        if self._fixed_descriptor is not None:
            return self._fixed_descriptor.describe_many(
                client_state_dicts,
                self.state_dict_for_clients(),
            )

        mode = str(getattr(self.config, "svdd_input_mode", "absolute")).lower().strip()
        reference_sd = self._state_dict_for_clients() if mode == "delta" else None
        return build_svdd_feature_matrix(
            client_state_dicts,
            self._svdd_feat,
            input_mode=mode,
            reference_state_dict=reference_sd,
        )

    def _state_dict_for_clients(self) -> Dict[str, Tensor]:
        """Return a detached CPU copy of global state_dict for broadcasting."""

        sd = self.global_model.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    def phase1_step(
        self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, Tensor, Tensor]:
        """Train the AE and select Phase-1 clients.

        The default ``phase1_selection='reconstruction'`` implements the
        intended attribution hypothesis: arbitrary parameter/gradient poisoning
        should reconstruct poorly and be excluded.  The old feature-median
        selector remains available as an explicit ablation.  AE training uses
        all clients in the default mode so that the first warm-up round does
        not make a decision from an untrained encoder; selection is performed
        from the post-update reconstruction errors.

        Returns:
            center_norm, z_variance, ae_loss,
            per-client reconstruction loss (all K, for monitoring), keep_mask (bool K)
        """

        X = self._build_input_matrix(client_state_dicts)  # (K, D_feat)
        X = robust_scale_features(X)
        K = int(X.shape[0])
        ratio = float(getattr(self.config, "ae_warmup_keep_ratio", 0.8))
        ratio = min(max(ratio, 1e-6), 1.0)
        num_keep = max(1, min(K, int(round(ratio * K))))

        selection = str(getattr(self.config, "phase1_selection", "reconstruction")).lower().strip()
        if selection not in {"reconstruction", "feature_median"}:
            raise ValueError(
                "phase1_selection must be 'reconstruction' or 'feature_median', "
                f"got {selection!r}."
            )
        X_dev = X.to(self.device)
        if selection == "feature_median":
            ref = X.median(dim=0).values
            selector_scores = torch.norm(X - ref.unsqueeze(0), dim=1)
            _, idx_keep = torch.topk(selector_scores, k=num_keep, largest=False)
            idx_keep = torch.sort(idx_keep).values
            X_train = X[idx_keep].to(self.device)
        else:
            # All clients contribute to representation learning. This is the
            # only defensible way to score the first round with a fresh AE.
            idx_keep = None
            X_train = X_dev

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = (x_hat - X_train).abs().sum(dim=1)
        loss = per_sample_loss.mean()

        self.optimizer_ae.zero_grad()
        loss.backward()
        clip_grad_norm_(self.ae.parameters(), self.config.ae_grad_clip)
        self.optimizer_ae.step()

        # Score after the update. For the reconstruction selector, low error
        # clients are the Phase-1 trusted set. For the legacy selector, retain
        # the feature-median decision while exposing reconstruction errors for
        # the attribution experiments.
        self.ae.eval()
        with torch.no_grad():
            x_hat_post = self.ae(X_dev)
            per_client_loss = (x_hat_post - X_dev).abs().sum(dim=1)
        if selection == "reconstruction":
            _, idx_keep = torch.topk(per_client_loss, k=num_keep, largest=False)
            idx_keep = torch.sort(idx_keep).values
        assert idx_keep is not None
        keep_mask = torch.zeros(K, dtype=torch.bool)
        keep_mask[idx_keep] = True

        selected_sds = [client_state_dicts[int(i)] for i in idx_keep.tolist()]
        alpha_sel = torch.full((num_keep,), 1.0 / float(num_keep))
        global_sd = weighted_fedavg(
            selected_sds,
            alpha_sel,
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        with torch.no_grad():
            Z = self.ae.encode(X.to(self.device))
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return center_norm, z_var, float(loss.item()), per_client_loss.detach().cpu(), keep_mask.detach().cpu()
    
    def init_center(self, client_state_dicts: List[Dict[str, Tensor]]) -> Tuple[float, float]:
        """Initialize SVDD center c using well-reconstructed clients."""

        X = self._build_input_matrix(client_state_dicts)
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

        X = self._build_input_matrix(client_state_dicts)
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
        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
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
                    ("Phase-1 Selector", str(self.config.phase1_selection)),
                    ("Feature Mode", str(self.config.svdd_feature_mode)),
                    (
                        "SVDD Input",
                        "delta" if self._fixed_descriptor is not None else str(self.config.svdd_input_mode),
                    ),
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
                ("Feature Mode", str(self.config.svdd_feature_mode)),
                (
                    "SVDD Input",
                    "delta" if self._fixed_descriptor is not None else str(self.config.svdd_input_mode),
                ),
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
    "alignins": AlignInsServer,
    "bnguard": BNGuardServer,
    "flgmm": FLGMMServer,
    "flanders": FLANDERSServer,
    "dmc": FedDMCServer,
}

# Backward compatibility
FederatedServer = SVDDServer
