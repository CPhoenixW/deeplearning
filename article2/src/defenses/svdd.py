from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
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


class ReconstructionObjective:
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).abs().mean(dim=1)


class SVDDObjective:
    def __call__(self, embedding: Tensor, center: Tensor) -> Tensor:
        return ((embedding - center) ** 2).sum(dim=1)


def _lower_quantile_mask(
    values: Tensor,
    quantile: float,
    *,
    parameter_name: str,
) -> Tensor:
    """Select finite values at or below a validated lower quantile."""

    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError(f"{parameter_name} must be in (0, 1].")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Quantile selection requires a non-empty 1D tensor.")
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("Quantile selection received non-finite values.")
    cutoff = torch.quantile(values.detach(), q)
    return values <= cutoff


class SVDDDefense(BaseDefense):
    """Two-phase AE-SVDD defense server."""

    defense_name = "svdd"
    # This is an internal protocol grid, not a user-facing hyperparameter.
    TOPK_REJECT_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50)

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
        validation_loader: DataLoader | None = None,
        svdd_feature_extractor: Optional[Callable[[Dict[str, Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(config, d_bn, device, model_fn, validation_loader)
        if not math.isfinite(float(config.alpha)) or not 0.0 <= float(config.alpha) <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1].")
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
                global_ratio=float(config.param_descriptor_global_ratio),
                layer_ratio=float(config.param_descriptor_layer_ratio),
                statistics_ratio=float(config.param_descriptor_statistics_ratio),
            )
        # 限制潜在空间维度，避免高维距离退化
        latent_dim = min(config.latent_dim, 64)
        self.ae = AutoEncoder(d_bn=d_bn, latent_dim=latent_dim).to(self.device)

        self.c: Optional[Tensor] = None

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )
        self.reconstruction_objective = ReconstructionObjective()
        self.svdd_objective = SVDDObjective()

    def _validation_accuracy(self) -> float:
        if self.validation_loader is None:
            return float("nan")
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for inputs, targets in self.validation_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.global_model(inputs)
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                total += int(targets.numel())
        return float(correct / max(1, total))

    def _select_topk_by_validation(
        self,
        scores: Tensor,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, float, float, Dict[str, float]]:
        """Choose a Top-K mask by clean validation accuracy.

        ``scores`` preserve the original phase-specific ranking: reconstruction
        loss in Phase 1 and SVDD distance in Phase 2.  Only the cutoff changes.
        """
        if scores.ndim != 1 or scores.numel() != len(client_state_dicts):
            raise ValueError("Top-K scores must match the client count.")
        finite = torch.isfinite(scores)
        if not bool(finite.any().item()):
            raise FloatingPointError("All client scores are non-finite.")
        ranked = torch.where(finite, scores, torch.full_like(scores, float("inf")))
        order = torch.argsort(ranked, stable=True)
        if self.validation_loader is None:
            # Direct unit-level defense tests may omit data. Production runs
            # always provide the fixed clean server validation loader.
            reject_ratio = 0.20
            keep_count = max(1, int(round((1.0 - reject_ratio) * len(client_state_dicts))))
            indices = order[:keep_count].detach().cpu()
            mask = torch.zeros(len(client_state_dicts), dtype=torch.bool)
            mask[indices] = True
            weights = mask.float() / float(keep_count)
            selected_state = weighted_fedavg(
                client_state_dicts, weights, device=self.aggregation_device
            )
            self.global_model.load_state_dict(selected_state)
            return mask.float(), weights, reject_ratio, float("nan"), {}
        candidate_accuracies: Dict[str, float] = {}
        best_mask: Tensor | None = None
        best_weights: Tensor | None = None
        best_ratio = self.TOPK_REJECT_RATIOS[0]
        best_accuracy = float("-inf")
        for reject_ratio in self.TOPK_REJECT_RATIOS:
            keep_count = max(1, min(len(client_state_dicts), int(round(
                (1.0 - reject_ratio) * len(client_state_dicts)
            ))))
            indices = order[:keep_count].detach().cpu()
            mask = torch.zeros(len(client_state_dicts), dtype=torch.bool)
            mask[indices] = True
            weights = mask.float() / float(keep_count)
            candidate_state = weighted_fedavg(
                client_state_dicts,
                weights,
                device=self.aggregation_device,
            )
            self.global_model.load_state_dict(candidate_state)
            accuracy = self._validation_accuracy()
            candidate_accuracies[f"{reject_ratio:.2f}"] = accuracy
            # Iterate from low to high so an equal-score update replaces the
            # previous candidate and ties choose the largest rejection ratio.
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_ratio = reject_ratio
                best_mask = mask
                best_weights = weights
        if best_mask is None or best_weights is None:
            raise RuntimeError("Top-K validation selection produced no candidate.")
        selected_state = weighted_fedavg(
            client_state_dicts,
            best_weights,
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(selected_state)
        return best_mask.float(), best_weights, best_ratio, best_accuracy, candidate_accuracies

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
    ) -> Tuple[float, float, float, Tensor, Tensor, float, float, Dict[str, float]]:
        """Train the AE and select the trusted Phase-1 clients.

        Train on all finite rows, rank clients by reconstruction error, and let
        the clean server validation set choose the Top-K cutoff.

        Returns:
            center_norm, z_variance, ae_loss,
            per-client reconstruction loss, keep mask, selected reject ratio,
            selected validation accuracy, candidate validation accuracies
        """

        raw_X = self._build_input_matrix(client_state_dicts)  # (K, D_feat)
        X, finite_rows = self._scale_input_matrix(raw_X)
        finite_indices = torch.where(finite_rows)[0]
        X_train = X[finite_indices].to(self.device)

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = self.reconstruction_objective(x_hat, X_train)
        loss = per_sample_loss.mean()

        self._safe_ae_step(loss, self.config.ae_grad_clip)

        self.ae.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            per_client_loss = self.reconstruction_objective(
                self.ae(X_device), X_device
            )
            per_client_loss = torch.where(
                finite_rows.to(self.device),
                per_client_loss,
                torch.full_like(per_client_loss, float("inf")),
            )
        keep_mask, weights, selected_ratio, validation_accuracy, candidates = (
            self._select_topk_by_validation(per_client_loss.detach().cpu(), client_state_dicts)
        )

        with torch.no_grad():
            self.ae.eval()
            Z = self.ae.encode(X.to(self.device))
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return (
            center_norm,
            z_var,
            float(loss.item()),
            per_client_loss.detach().cpu(),
            keep_mask.detach().cpu(),
            float(selected_ratio),
            float(validation_accuracy),
            candidates,
        )
    
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
            finite_indices = torch.where(finite_recon)[0]
            selected = _lower_quantile_mask(
                recon_error[finite_recon],
                self.config.center_init_quantile,
                parameter_name="center_init_quantile",
            )
            c = Z[finite_indices[selected]].mean(dim=0)
            if not bool(torch.isfinite(c).all().item()):
                raise FloatingPointError("SVDD center initialization produced non-finite values.")
            c[c.abs() < 0.01] = 0.01

        self.c = c.detach()
        center_norm = float(self.c.norm().item())
        z_var = float(Z.var().item())
        return center_norm, z_var

    def phase2_step(
        self, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[
        float,
        float,
        float,
        float,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        float,
        float,
        Dict[str, float],
    ]:
        """Run one SVDD-filtered aggregation round.

        Returns:
            center_norm, z_variance, svdd_loss_value, recon_loss_value,
            d, M, alpha, per-client reconstruction loss, selected reject ratio,
            validation accuracy, candidate validation accuracies
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
        M, alpha, selected_ratio, validation_accuracy, candidates = (
            self._select_topk_by_validation(d.detach().cpu(), client_state_dicts)
        )

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
        keep = _lower_quantile_mask(
            recon_per_sample,
            self.config.phase2_recon_quantile,
            parameter_name="phase2_recon_quantile",
        )
        recon_loss = recon_per_sample[keep].mean()

        total_loss = self.config.alpha * svdd_loss + (1.0 - self.config.alpha) * recon_loss

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
            d.detach().cpu(),
            M.detach().cpu(),
            alpha.detach().cpu(),
            recon_per_client,
            float(selected_ratio),
            float(validation_accuracy),
            candidates,
        )

    def _phase1_result(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        (
            center_norm,
            z_var,
            ae_loss,
            scores,
            keep_mask,
            selected_ratio,
            validation_accuracy,
            candidates,
        ) = self.phase1_step(round_idx, client_state_dicts)
        weights = keep_mask / (keep_mask.sum() + 1e-12)
        kept = int(keep_mask.sum().item())
        total = len(client_state_dicts)
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=ae_loss,
            svdd_loss=float("nan"),
            d=scores,
            m=keep_mask,
            alpha=weights,
            phase="warmup",
            show_detection=True,
            monitor_items=[
                ("Defense", "SVDD"),
                ("Feature Mode", str(self.config.svdd_feature_mode)),
                ("Kept clients", f"{kept}/{total}"),
                ("Selected reject ratio", f"{selected_ratio:.2f}"),
                ("Validation accuracy", f"{validation_accuracy:.6f}"),
                ("Center L2-Norm", f"{center_norm:.6f}"),
                ("Z-Space Variance", f"{z_var:.6f}"),
                ("AE L1-Loss", f"{ae_loss:.6f}"),
            ],
            recon_loss=ae_loss,
            total_loss=ae_loss,
            server_metrics={
                "selected_reject_ratio": selected_ratio,
                "validation_accuracy": validation_accuracy,
                "validation_candidates": candidates,
            },
            participant_metrics={"reconstruction_loss": scores.detach().cpu()},
        )

    def _phase2_result(
        self,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        if self.c is None:
            self.init_center(client_state_dicts)
        (
            center_norm,
            z_var,
            svdd_loss,
            recon_loss,
            scores,
            accepted,
            weights,
            recon_per_client,
            selected_ratio,
            validation_accuracy,
            candidates,
        ) = self.phase2_step(client_state_dicts)
        kept = int(accepted.sum().item())
        total = len(client_state_dicts)
        total_loss = self.config.alpha * svdd_loss + (1.0 - self.config.alpha) * recon_loss
        total_per_client = (
            self.config.alpha * scores
            + (1.0 - self.config.alpha) * recon_per_client
        )
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=float("nan"),
            svdd_loss=svdd_loss,
            recon_loss=recon_loss,
            total_loss=total_loss,
            d=scores,
            m=accepted,
            alpha=weights,
            phase="filtering",
            show_detection=True,
            monitor_items=[
                ("Defense", "SVDD"),
                ("Feature Mode", str(self.config.svdd_feature_mode)),
                ("Kept clients", f"{kept}/{total}"),
                ("Selected reject ratio", f"{selected_ratio:.2f}"),
                ("Validation accuracy", f"{validation_accuracy:.6f}"),
                ("Center L2-Norm", f"{center_norm:.6f}"),
                ("Z-Space Variance", f"{z_var:.6f}"),
            ],
            server_metrics={
                "selected_reject_ratio": selected_ratio,
                "validation_accuracy": validation_accuracy,
                "validation_candidates": candidates,
            },
            participant_metrics={
                "svdd_loss": scores.detach().cpu(),
                "reconstruction_loss": recon_per_client.detach().cpu(),
                "total_loss": total_per_client.detach().cpu(),
            },
        )

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        if round_idx <= int(self.config.phase1_rounds):
            return self._phase1_result(round_idx, client_state_dicts)
        return self._phase2_result(client_state_dicts)


__all__ = ["SVDDDefense", "_lower_quantile_mask"]
