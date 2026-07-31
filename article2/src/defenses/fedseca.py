from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from ..config import FedConfig
from .base import BaseDefense, DefenseResult as RoundStats
from .common import (
    _aggregate_nonparam_buffers,
    _apply_flat_param_delta_to_global,
    _flatten_param_delta,
)


class FedSECADefense(BaseDefense):
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

    def _aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
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
            participant_metrics={
                "concordance": omega_scores.detach().cpu(),
                "cosine_similarity": cos_scores.detach().cpu(),
            },
        )


__all__ = ["FedSECADefense"]
