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
    TOPK_REJECT_RATIOS = (0.0, 0.10, 0.20, 0.30, 0.40)
    SCORE_MODES = ("legacy", "recon", "combined", "svdd")

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
        if not math.isfinite(float(config.svdd_lambda)) or not 0.0 <= float(config.svdd_lambda) <= 1.0:
            raise ValueError("svdd_lambda must be finite and in [0, 1].")
        legacy_mode = str(getattr(config, "svdd_score_mode", "legacy") or "legacy").lower().strip()
        phase1_mode = getattr(config, "phase1_score_mode", None)
        phase2_mode = getattr(config, "phase2_score_mode", None)
        if phase1_mode is None and phase2_mode is None:
            if legacy_mode == "legacy":
                phase1_mode, phase2_mode = "recon", "svdd"
            else:
                # Preserve old sensitivity configs that intentionally applied
                # one score mode to both phases.
                phase1_mode, phase2_mode = legacy_mode, legacy_mode
        else:
            phase1_mode = phase1_mode or "recon"
            phase2_mode = phase2_mode or ("svdd" if legacy_mode == "legacy" else legacy_mode)
        self.phase1_score_mode = self._normalize_score_mode(phase1_mode, "phase1_score_mode")
        self.phase2_score_mode = self._normalize_score_mode(phase2_mode, "phase2_score_mode")
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
        self.center_shift = 0.0

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )
        self.reconstruction_objective = ReconstructionObjective()
        self.svdd_objective = SVDDObjective()

    @classmethod
    def _normalize_score_mode(cls, value: object, field_name: str) -> str:
        mode = str(value).lower().strip()
        if mode == "legacy":
            raise ValueError(f"{field_name} must resolve to recon, combined, or svdd.")
        if mode not in cls.SCORE_MODES:
            raise ValueError(
                f"{field_name} must be one of {tuple(item for item in cls.SCORE_MODES if item != 'legacy')}, got {mode!r}."
            )
        return mode

    @staticmethod
    def _rank_score(values: Tensor) -> Tensor:
        """Map finite anomaly values to [0, 1] ranks, preserving invalid rows."""

        values = values.float()
        finite = torch.isfinite(values)
        result = torch.full_like(values, float("inf"))
        indices = torch.where(finite)[0]
        if indices.numel() == 0:
            return result
        order = indices[torch.argsort(values[indices], stable=True)]
        if order.numel() == 1:
            result[order] = 0.0
        else:
            result[order] = torch.arange(
                order.numel(), device=values.device, dtype=values.dtype
            ) / float(order.numel() - 1)
        return result

    def _selection_score(
        self,
        reconstruction: Tensor,
        svdd: Tensor,
        *,
        phase: str,
    ) -> Tensor:
        """Build the phase-specific score used by validation Top-K selection."""

        mode = self.phase1_score_mode if phase == "phase1" else self.phase2_score_mode
        if mode == "recon":
            score = reconstruction
        elif mode == "svdd":
            score = svdd
        else:  # combined
            recon_rank = self._rank_score(reconstruction)
            svdd_rank = self._rank_score(svdd)
            score = 0.5 * (recon_rank + svdd_rank)
        valid = torch.isfinite(reconstruction) & torch.isfinite(svdd)
        return torch.where(valid, score, torch.full_like(score, float("inf")))

    def _phase1_svdd_proxy(self, embeddings: Tensor, finite_rows: Tensor) -> Tensor:
        """Use the current embedding median as a provisional SVDD center."""

        proxy = torch.full(
            (embeddings.shape[0],), float("inf"), device=embeddings.device
        )
        if bool(finite_rows.any().item()):
            center = embeddings[finite_rows].median(dim=0).values
            proxy[finite_rows] = ((embeddings[finite_rows] - center) ** 2).sum(dim=1)
        return proxy

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

        ``scores`` preserve the configured phase-specific ranking (reconstruction,
        SVDD distance, or their combined rank score). Only the cutoff changes.
        """
        if scores.ndim != 1 or scores.numel() != len(client_state_dicts):
            raise ValueError("Top-K scores must match the client count.")
        finite = torch.isfinite(scores)
        if not bool(finite.any().item()):
            raise FloatingPointError("All client scores are non-finite.")
        ranked = torch.where(finite, scores, torch.full_like(scores, float("inf")))
        order = torch.argsort(ranked, stable=True)
        # Non-finite feature/score rows are never eligible for Top-K, even
        # when the requested rejection ratio would otherwise keep more than
        # the finite population.  This prevents invalid uploads from being
        # assigned positive aggregation weight and makes the effective filter
        # stricter than the nominal 50% cap when necessary.
        finite_order = order[finite[order]]
        if self.validation_loader is None:
            # Direct unit-level defense tests may omit data. Production runs
            # always provide the fixed clean server validation loader.
            reject_ratio = 0.20
            keep_count = max(
                1,
                min(
                    int(finite_order.numel()),
                    int(round((1.0 - reject_ratio) * len(client_state_dicts))),
                ),
            )
            indices = finite_order[:keep_count].detach().cpu()
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
            keep_count = max(
                1,
                min(
                    int(finite_order.numel()),
                    int(round((1.0 - reject_ratio) * len(client_state_dicts))),
                ),
            )
            indices = finite_order[:keep_count].detach().cpu()
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
    ) -> Tuple[
        float,
        float,
        float,
        Tensor,
        Tensor,
        Tensor,
        float,
        float,
        Dict[str, float],
    ]:
        """Train the AE and select the trusted Phase-1 clients.

        Train on all finite rows, rank clients by reconstruction error, and let
        the clean server validation set choose the Top-K cutoff.

        Returns:
            center_norm, z_variance, ae_loss,
            selection score, per-client reconstruction loss, keep mask,
            selected reject ratio, selected validation accuracy, candidate
            validation accuracies
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
            embeddings = self.ae.encode(X_device)
            svdd_proxy = self._phase1_svdd_proxy(
                embeddings, finite_rows.to(self.device)
            )
            selection_scores = self._selection_score(
                per_client_loss,
                svdd_proxy,
                phase="phase1",
            )
        keep_mask, weights, selected_ratio, validation_accuracy, candidates = (
            self._select_topk_by_validation(
                selection_scores.detach().cpu(), client_state_dicts
            )
        )

        with torch.no_grad():
            self.ae.eval()
            Z = embeddings
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return (
            center_norm,
            z_var,
            float(loss.item()),
            selection_scores.detach().cpu(),
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
            svdd_proxy = self._phase1_svdd_proxy(Z, finite_recon)
            selection_scores = self._selection_score(
                recon_error,
                svdd_proxy,
                phase="phase1",
            )
            finite_indices = torch.where(finite_recon)[0]
            selected = _lower_quantile_mask(
                selection_scores[finite_recon],
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
        Tensor,
        float,
        float,
        Dict[str, float],
    ]:
        """Run one SVDD-filtered aggregation round.

        Returns:
            center_norm, z_variance, svdd_loss_value, recon_loss_value,
            d, selection score, M, aggregation weights, per-client reconstruction loss,
            selected reject ratio, validation accuracy, candidate validation
            accuracies
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
        selection_scores = self._selection_score(
            recon_per_client.to(self.device),
            d,
            phase="phase2",
        )
        accepted_mask, aggregation_weights, selected_ratio, validation_accuracy, candidates = (
            self._select_topk_by_validation(
                selection_scores.detach().cpu(), client_state_dicts
            )
        )

        trusted = accepted_mask > 0.5
        self.center_shift = 0.0
        if trusted.sum() > 0:
            with torch.no_grad():
                c_new = Z[trusted].mean(dim=0)
                c_updated = self.config.center_ema_decay * c + (1.0 - self.config.center_ema_decay) * c_new
                c_updated[c_updated.abs() < 0.01] = 0.01
                self.center_shift = float((c_updated - c).norm().item())
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

        total_loss = (
            self.config.svdd_lambda * svdd_loss
            + (1.0 - self.config.svdd_lambda) * recon_loss
        )

        self._safe_ae_step(total_loss, self.config.svdd_grad_clip)

        # Weighted aggregation
        global_sd = weighted_fedavg(
            client_state_dicts,
            aggregation_weights.detach().cpu(),
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
            selection_scores.detach().cpu(),
            accepted_mask.detach().cpu(),
            aggregation_weights.detach().cpu(),
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
            recon_scores,
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
                ("Phase 1 Score", self.phase1_score_mode),
                ("Phase 2 Score", self.phase2_score_mode),
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
                "center_shift": 0.0,
                "accepted_client_ids": torch.where(keep_mask > 0.5)[0].tolist(),
                "aggregation_weights": weights.tolist(),
            },
            participant_metrics={
                "selection_score": scores.detach().cpu(),
                "reconstruction_loss": recon_scores.detach().cpu(),
            },
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
            svdd_scores,
            selection_scores,
            accepted,
            aggregation_weights,
            recon_per_client,
            selected_ratio,
            validation_accuracy,
            candidates,
        ) = self.phase2_step(client_state_dicts)
        kept = int(accepted.sum().item())
        total = len(client_state_dicts)
        total_loss = (
            self.config.svdd_lambda * svdd_loss
            + (1.0 - self.config.svdd_lambda) * recon_loss
        )
        total_per_client = (
            self.config.svdd_lambda * svdd_scores
            + (1.0 - self.config.svdd_lambda) * recon_per_client
        )
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=float("nan"),
            svdd_loss=svdd_loss,
            recon_loss=recon_loss,
            total_loss=total_loss,
            d=selection_scores,
            m=accepted,
            alpha=aggregation_weights,
            phase="filtering",
            show_detection=True,
            monitor_items=[
                ("Defense", "SVDD"),
                ("Feature Mode", str(self.config.svdd_feature_mode)),
                ("Phase 1 Score", self.phase1_score_mode),
                ("Phase 2 Score", self.phase2_score_mode),
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
                "center_shift": self.center_shift,
                "accepted_client_ids": torch.where(accepted > 0.5)[0].tolist(),
                "aggregation_weights": aggregation_weights.tolist(),
            },
            participant_metrics={
                "svdd_loss": svdd_scores.detach().cpu(),
                "reconstruction_loss": recon_per_client.detach().cpu(),
                "selection_score": selection_scores.detach().cpu(),
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
