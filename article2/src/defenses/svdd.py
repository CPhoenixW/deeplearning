from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from ..config import FedConfig
from ..fixed_descriptor import FixedHierarchicalMultiViewDescriptor
from ..models import AutoEncoder
from ..utils import (
    build_svdd_feature_matrix,
    extract_bn_features,
    robust_scale_features,
    weighted_fedavg,
)
from .base import BaseDefense, DefenseResult as RoundStats
from .svdd_phases import (
    FilteringPhase,
    ReconstructionObjective,
    SVDDObjective,
    WarmupPhase,
)


def _compute_svdd_keep_mask(
    d: Tensor,
    svdd_round: int,
    config: FedConfig,
) -> Tuple[Tensor, float, float]:
    """Apply the scheduled MAD threshold and an optional closest-client cap."""

    if d.ndim != 1 or d.numel() == 0:
        raise ValueError("SVDD distances must be a non-empty 1D tensor.")
    finite = torch.isfinite(d)
    if not bool(finite.any().item()):
        raise FloatingPointError("All SVDD client distances are non-finite.")

    finite_d = d[finite]
    med_d = torch.median(finite_d)
    mad_d = 1.4826 * torch.median((finite_d - med_d).abs())
    mad_d = torch.clamp(mad_d, min=1e-6)
    warmup_rounds = max(1, int(config.svdd_warmup_rounds))
    p_tau = min(1.0, max(0.0, svdd_round / float(warmup_rounds)))
    if config.tau_start > 0.0 and config.tau_end > 0.0:
        tau = config.tau_start - p_tau * (config.tau_start - config.tau_end)
    else:
        tau = config.tau_multiplier
    threshold = med_d + tau * mad_d

    keep_mask = finite & (d <= threshold)
    max_keep_ratio = float(getattr(config, "svdd_max_keep_ratio", 1.0))
    if not 0.0 < max_keep_ratio <= 1.0:
        raise ValueError("svdd_max_keep_ratio must be in (0, 1].")
    max_keep = max(1, min(d.numel(), int(round(max_keep_ratio * d.numel()))))
    if int(keep_mask.sum().item()) > max_keep:
        ranked = torch.where(finite, d, torch.full_like(d, float("inf")))
        idx_keep = torch.topk(ranked, k=max_keep, largest=False).indices
        keep_mask = torch.zeros_like(finite)
        keep_mask[idx_keep] = True

    if not bool(keep_mask.any().item()):
        ranked = torch.where(finite, d, torch.full_like(d, float("inf")))
        keep_mask[torch.argmin(ranked)] = True

    return keep_mask.float(), float(tau), float(threshold.item())


class SVDDDefense(BaseDefense):
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
        self.warmup_phase = WarmupPhase()
        self.filtering_phase = FilteringPhase()
        self.reconstruction_objective = ReconstructionObjective()
        self.svdd_objective = SVDDObjective()

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

    def _scale_input_matrix(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        """Sanitize and clip AE inputs while preserving invalid-row identity."""

        finite_rows = torch.isfinite(raw).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All SVDD feature rows are non-finite.")
        clip_value = float(getattr(self.config, "svdd_feature_clip", 10.0))
        scaled = robust_scale_features(raw, clip_value=clip_value)
        if not bool(torch.isfinite(scaled).all().item()):
            raise FloatingPointError("SVDD feature scaling produced non-finite values.")
        return scaled, finite_rows

    def _safe_ae_step(self, loss: Tensor, max_grad_norm: float) -> bool:
        """Apply one AE step without allowing non-finite state to persist."""

        parameters = [p for p in self.ae.parameters() if p.requires_grad]
        self.optimizer_ae.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss.detach()).item()):
            return False

        loss.backward()
        try:
            grad_norm = clip_grad_norm_(
                parameters,
                float(max_grad_norm),
                error_if_nonfinite=True,
            )
        except RuntimeError:
            self.optimizer_ae.zero_grad(set_to_none=True)
            return False

        gradients_finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item())
            for p in parameters
        )
        if not bool(torch.isfinite(grad_norm).item()) or not gradients_finite:
            self.optimizer_ae.zero_grad(set_to_none=True)
            return False

        parameter_backup = [p.detach().clone() for p in parameters]
        self.optimizer_ae.step()
        if all(bool(torch.isfinite(p).all().item()) for p in parameters):
            return True

        # This should be rare after finite gradient clipping. Restore the last
        # valid parameters and clear Adam moments so a bad state cannot leak
        # into the next communication round.
        with torch.no_grad():
            for parameter, backup in zip(parameters, parameter_backup):
                parameter.copy_(backup)
        self.optimizer_ae.state.clear()
        self.optimizer_ae.zero_grad(set_to_none=True)
        return False

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

        raw_X = self._build_input_matrix(client_state_dicts)  # (K, D_feat)
        X, finite_rows = self._scale_input_matrix(raw_X)
        K = int(X.shape[0])
        ratio = float(getattr(self.config, "ae_warmup_keep_ratio", 0.8))
        ratio = min(max(ratio, 1e-6), 1.0)
        num_finite = int(finite_rows.sum().item())
        num_keep = max(1, min(num_finite, int(round(ratio * K))))

        ref = X.median(dim=0).values
        distances = torch.norm(X - ref.unsqueeze(0), dim=1)
        distances = torch.where(
            finite_rows,
            distances,
            torch.full_like(distances, float("inf")),
        )
        _, idx_keep = torch.topk(distances, k=num_keep, largest=False)
        idx_keep = torch.sort(idx_keep).values

        keep_mask = torch.zeros(K, dtype=torch.bool)
        keep_mask[idx_keep] = True

        self.ae.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            x_hat_cur = self.ae(X_dev)
            per_client_loss = self.reconstruction_objective(x_hat_cur, X_dev)
            per_client_loss = torch.where(
                finite_rows.to(self.device),
                per_client_loss,
                torch.full_like(per_client_loss, float("inf")),
            )

        X_train = X[idx_keep].to(self.device)

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = self.reconstruction_objective(x_hat, X_train)
        loss = per_sample_loss.mean()

        self._safe_ae_step(loss, self.config.ae_grad_clip)

        selected_sds = [client_state_dicts[int(i)] for i in idx_keep.tolist()]
        alpha_sel = torch.full((num_keep,), 1.0 / float(num_keep))
        global_sd = weighted_fedavg(
            selected_sds,
            alpha_sel,
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        with torch.no_grad():
            self.ae.eval()
            Z = self.ae.encode(X.to(self.device))
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return center_norm, z_var, float(loss.item()), per_client_loss.detach().cpu(), keep_mask.detach().cpu()
    
    def init_center(self, client_state_dicts: List[Dict[str, Tensor]]) -> Tuple[float, float]:
        """Initialize SVDD center c using well-reconstructed clients."""

        raw_X = self._build_input_matrix(client_state_dicts)
        X, finite_rows = self._scale_input_matrix(raw_X)
        X = X.to(self.device)
        finite_rows = finite_rows.to(self.device)
        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X)
            x_hat = self.ae(X)
            recon_error = (x_hat - X).abs().sum(dim=1)
            finite_recon = finite_rows & torch.isfinite(recon_error)
            if not bool(finite_recon.any().item()):
                raise FloatingPointError("No finite clients are available to initialize SVDD center.")
            med = torch.median(recon_error[finite_recon])
            init_mask = finite_recon & (recon_error <= med)
            c = Z[init_mask].mean(dim=0)
            if not bool(torch.isfinite(c).all().item()):
                raise FloatingPointError("SVDD center initialization produced non-finite values.")
            c[c.abs() < 0.01] = 0.01

        self.c = c.detach()
        center_norm = float(self.c.norm().item())
        z_var = float(Z.var().item())
        return center_norm, z_var

    def phase2_step(
        self, svdd_round: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor]:
        """Run one SVDD-filtered aggregation round.

        Returns:
            center_norm, z_variance, svdd_loss_value, recon_loss_value,
            tau, threshold, d, M, alpha, per-client reconstruction loss
        """

        assert self.c is not None, "SVDD center c must be initialized before Phase 2."

        raw_X = self._build_input_matrix(client_state_dicts)
        X, finite_feature_rows = self._scale_input_matrix(raw_X)

        # Embeddings without grad
        self.ae.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            Z = self.ae.encode(X_device)
            # Keep one reconstruction value per original client for the
            # defense-owned reporter. The scalar ``recon_loss`` below remains
            # the optimization objective after robust client filtering.
            recon_per_client = self.reconstruction_objective(
                self.ae(X_device), X_device
            ).detach().cpu()
            recon_per_client = torch.where(
                finite_feature_rows,
                recon_per_client,
                torch.full_like(recon_per_client, float("inf")),
            )
        c = self.c.to(self.device)
        d = self.svdd_objective(Z, c)
        d = torch.where(
            finite_feature_rows.to(self.device),
            d,
            torch.full_like(d, float("inf")),
        )
        M, tau, threshold = _compute_svdd_keep_mask(d, svdd_round, self.config)

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
        svdd_loss = self.svdd_objective(
            Z_trusted, self.c.detach().to(self.device)
        ).mean()

        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = True

        finite_rows = finite_feature_rows & torch.isfinite(X).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All SVDD feature rows are non-finite.")
        X_cur = X[finite_rows].to(self.device)
        X_cur_hat = self.ae(X_cur)
        recon_per_sample = self.reconstruction_objective(X_cur_hat, X_cur)
        q80 = torch.quantile(recon_per_sample.detach(), 0.8)
        keep = recon_per_sample <= q80
        recon_loss = recon_per_sample[keep].mean()

        total_loss = (
            self.config.svdd_loss_weight * svdd_loss
            + self.config.recon_loss_weight * recon_loss
        )

        self._safe_ae_step(total_loss, self.config.svdd_grad_clip)

        # Weighted aggregation
        global_sd = weighted_fedavg(
            client_state_dicts,
            alpha.detach().cpu(),
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(global_sd)

        center_norm = float(self.c.norm().item())
        finite_z = Z[torch.isfinite(Z).all(dim=1)]
        z_var = float(finite_z.var().item()) if finite_z.numel() > 1 else 0.0

        return (
            center_norm,
            z_var,
            float(svdd_loss.item()),
            float(recon_loss.item()),
            tau,
            threshold,
            d.detach().cpu(),
            M.detach().cpu(),
            alpha.detach().cpu(),
            recon_per_client,
        )

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        if round_idx <= int(self.config.phase1_rounds):
            return self.warmup_phase.run(self, round_idx, client_state_dicts)
        return self.filtering_phase.run(self, round_idx, client_state_dicts)


__all__ = ["SVDDDefense", "_compute_svdd_keep_mask"]
