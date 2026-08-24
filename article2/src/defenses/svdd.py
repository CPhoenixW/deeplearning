from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from ..config import FedConfig
from ..models import AutoEncoder
from ..utils import (
    weighted_fedavg,
)
from .base import BaseDefense, DefenseResult as RoundStats


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
    SCORE_MODES = ("legacy", "recon", "combined", "svdd")

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
        validation_loader: DataLoader | None = None,
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
        self.input_mode = str(config.svdd_input_mode).lower().strip()
        if self.input_mode not in {"absolute", "delta"}:
            raise ValueError("svdd_input_mode must be 'absolute' or 'delta'.")
        self.input_dim = int(config.svdd_input_dim)
        if self.input_dim != 4096:
            raise ValueError("svdd_input_dim is fixed at 4096 for the unified SVDD protocol.")
        self.normalization_eps = float(config.svdd_normalization_eps)
        if not math.isfinite(self.normalization_eps) or self.normalization_eps <= 0.0:
            raise ValueError("svdd_normalization_eps must be positive and finite.")
        self._parameter_indices = self._build_parameter_indices()
        # Keep the configured latent dimension so the sensitivity sweep can
        # evaluate both compressed and overcomplete representations.
        latent_dim = int(config.latent_dim)
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        self.ae = AutoEncoder(d_bn=d_bn, latent_dim=latent_dim).to(self.device)

        self.c: Optional[Tensor] = None
        self.center_shift = 0.0

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )

    def _build_parameter_indices(self) -> Tensor:
        """Select exactly 4096 deterministic, layer-balanced parameter coordinates."""

        sizes = [
            int(parameter.numel())
            for name, parameter in self.global_model.named_parameters()
            if name in self.param_names and parameter.requires_grad
        ]
        total = sum(sizes)
        if total < self.input_dim:
            raise ValueError(
                f"The model has only {total:,} trainable parameters; "
                f"at least {self.input_dim:,} are required."
            )

        raw_quotas = [self.input_dim * size / total for size in sizes]
        quotas = [min(size, int(quota)) for size, quota in zip(sizes, raw_quotas)]
        remainder = self.input_dim - sum(quotas)
        order = sorted(
            range(len(sizes)),
            key=lambda index: raw_quotas[index] - quotas[index],
            reverse=True,
        )
        for index in order:
            if remainder == 0:
                break
            if quotas[index] < sizes[index]:
                quotas[index] += 1
                remainder -= 1
        if remainder:
            raise RuntimeError("Could not allocate the fixed SVDD input dimension.")

        indices: list[Tensor] = []
        offset = 0
        for size, quota in zip(sizes, quotas):
            if quota:
                if quota == size:
                    local = torch.arange(size, dtype=torch.long)
                elif quota == 1:
                    local = torch.tensor([(size - 1) // 2], dtype=torch.long)
                else:
                    local = torch.linspace(0, size - 1, quota).round().long()
                indices.append(local + offset)
            offset += size
        result = torch.cat(indices)
        if result.numel() != self.input_dim or result.unique().numel() != self.input_dim:
            raise RuntimeError("SVDD parameter coordinate selection is not unique.")
        return result

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
    def _reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).abs().mean(dim=1)

    @staticmethod
    def _svdd_loss(embedding: Tensor, center: Tensor) -> Tensor:
        return ((embedding - center) ** 2).sum(dim=1)

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
        if scores.ndim != 1 or scores.numel() != len(client_state_dicts):
            raise ValueError("Top-K scores must match the client count.")
        finite = torch.isfinite(scores)
        if not bool(finite.any().item()):
            raise FloatingPointError("All client scores are non-finite.")
        ranked = torch.where(finite, scores, torch.full_like(scores, float("inf")))
        order = torch.argsort(ranked, stable=True)
        finite_order = order[finite[order]]
        if self.validation_loader is None:
            reject_ratio = 0.20
            mask, weights = self._topk_candidate(
                finite_order, reject_ratio, len(client_state_dicts)
            )
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
            mask, weights = self._topk_candidate(
                finite_order, reject_ratio, len(client_state_dicts)
            )
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

    @staticmethod
    def _topk_candidate(
        finite_order: Tensor, reject_ratio: float, total_clients: int
    ) -> tuple[Tensor, Tensor]:
        keep = max(1, min(int(finite_order.numel()), int(round((1.0 - reject_ratio) * total_clients))))
        mask = torch.zeros(total_clients, dtype=torch.bool)
        mask[finite_order[:keep].detach().cpu()] = True
        return mask, mask.float() / float(keep)

    def _build_input_matrix(
        self, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tensor:
        """Build a fixed-width absolute-parameter or parameter-delta matrix."""

        reference = self.state_dict_for_clients() if self.input_mode == "delta" else None
        rows: list[Tensor] = []
        for state_dict in client_state_dicts:
            parts: list[Tensor] = []
            for name in self.param_names:
                value = state_dict[name]
                if not value.is_floating_point():
                    raise TypeError(f"Trainable parameter {name!r} is not floating point.")
                current = value.detach().cpu().float().reshape(-1)
                if reference is not None:
                    current = current - reference[name].detach().cpu().float().reshape(-1)
                parts.append(current)
            if not parts:
                raise ValueError("The model has no floating-point trainable parameters.")
            rows.append(torch.cat(parts))
        if not rows:
            raise ValueError("SVDD requires at least one client parameter state.")
        full = torch.stack(rows, dim=0)
        return full.index_select(1, self._parameter_indices)

    def _scale_input_matrix(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply finite-client feature-wise mean/std normalization."""

        finite_rows = torch.isfinite(raw).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All SVDD feature rows are non-finite.")
        safe = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = safe[finite_rows]
        mean = valid.mean(dim=0)
        std = valid.std(dim=0, unbiased=False).clamp_min(self.normalization_eps)
        scaled = (safe - mean) / std
        scaled[~finite_rows] = 0.0
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

    def phase1_step(self, client_state_dicts: List[Dict[str, Tensor]]) -> tuple:

        raw_X = self._build_input_matrix(client_state_dicts)  # (K, D_feat)
        X, finite_rows = self._scale_input_matrix(raw_X)
        finite_indices = torch.where(finite_rows)[0]
        X_train = X[finite_indices].to(self.device)

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = self._reconstruction_loss(x_hat, X_train)
        loss = per_sample_loss.mean()

        self._safe_ae_step(loss, self.config.ae_grad_clip)

        self.ae.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            per_client_loss = self._reconstruction_loss(
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

    def phase2_step(self, client_state_dicts: List[Dict[str, Tensor]]) -> tuple:

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
            recon_per_client = self._reconstruction_loss(
                self.ae(X_device), X_device
            ).detach().cpu()
            recon_per_client = torch.where(
                finite_feature_rows,
                recon_per_client,
                torch.full_like(recon_per_client, float("inf")),
            )
        c = self.c.to(self.device)
        d = self._svdd_loss(Z, c)
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
        svdd_loss = self._svdd_loss(
            Z_trusted, self.c.detach().to(self.device)
        ).mean()

        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = True

        finite_rows = finite_feature_rows & torch.isfinite(X).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All SVDD feature rows are non-finite.")
        X_cur = X[finite_rows].to(self.device)
        X_cur_hat = self.ae(X_cur)
        recon_per_sample = self._reconstruction_loss(X_cur_hat, X_cur)
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
            center_norm, z_var, float(svdd_loss.item()), float(recon_loss.item()), d.detach().cpu(),
            selection_scores.detach().cpu(), accepted_mask.detach().cpu(), aggregation_weights.detach().cpu(),
            recon_per_client, float(selected_ratio), float(validation_accuracy), candidates,
        )

    def _make_result(
        self,
        *,
        phase: str,
        center_norm: float,
        z_var: float,
        ae_loss: float,
        svdd_loss: float,
        recon_loss: float,
        total_loss: float,
        scores: Tensor,
        accepted: Tensor,
        weights: Tensor,
        selected_ratio: float,
        validation_accuracy: float,
        candidates: Dict[str, float],
        participant_metrics: Dict[str, Tensor],
    ) -> RoundStats:
        kept, total = int(accepted.sum().item()), len(accepted)
        monitor = [
            ("Defense", "SVDD"),
            ("Input Mode", self.input_mode),
            ("Phase 1 Score", self.phase1_score_mode),
            ("Phase 2 Score", self.phase2_score_mode),
            ("Kept clients", f"{kept}/{total}"),
            ("Selected reject ratio", f"{selected_ratio:.2f}"),
            ("Validation accuracy", f"{validation_accuracy:.6f}"),
            ("Center L2-Norm", f"{center_norm:.6f}"),
            ("Z-Space Variance", f"{z_var:.6f}"),
        ]
        if math.isfinite(ae_loss):
            monitor.append(("AE L1-Loss", f"{ae_loss:.6f}"))
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=ae_loss,
            svdd_loss=svdd_loss,
            recon_loss=recon_loss,
            total_loss=total_loss,
            d=scores,
            m=accepted,
            alpha=weights,
            phase=phase,
            show_detection=True,
            monitor_items=monitor,
            server_metrics={
                "selected_reject_ratio": selected_ratio,
                "validation_accuracy": validation_accuracy,
                "validation_candidates": candidates,
                "center_shift": self.center_shift if phase == "filtering" else 0.0,
                "accepted_client_ids": torch.where(accepted > 0.5)[0].tolist(),
                "aggregation_weights": weights.tolist(),
            },
            participant_metrics=participant_metrics,
        )

    def _phase1_result(self, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        center, variance, loss, scores, recon, accepted, ratio, accuracy, candidates = (
            self.phase1_step(client_state_dicts)
        )
        weights = accepted / accepted.sum().clamp_min(1.0)
        return self._make_result(
            phase="warmup", center_norm=center, z_var=variance,
            ae_loss=loss, svdd_loss=float("nan"), recon_loss=loss, total_loss=loss,
            scores=scores, accepted=accepted, weights=weights,
            selected_ratio=ratio, validation_accuracy=accuracy, candidates=candidates,
            participant_metrics={"selection_score": scores, "reconstruction_loss": recon},
        )

    def _phase2_result(self, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        if self.c is None:
            self.init_center(client_state_dicts)
        (
            center, variance, svdd_loss, recon_loss, svdd_scores, scores,
            accepted, weights, recon, ratio, accuracy, candidates,
        ) = self.phase2_step(client_state_dicts)
        total_loss = self.config.svdd_lambda * svdd_loss + (1.0 - self.config.svdd_lambda) * recon_loss
        return self._make_result(
            phase="filtering", center_norm=center, z_var=variance,
            ae_loss=float("nan"), svdd_loss=svdd_loss, recon_loss=recon_loss,
            total_loss=total_loss, scores=scores, accepted=accepted, weights=weights,
            selected_ratio=ratio, validation_accuracy=accuracy, candidates=candidates,
            participant_metrics={
                "svdd_loss": svdd_scores, "reconstruction_loss": recon,
                "selection_score": scores,
                "total_loss": self.config.svdd_lambda * svdd_scores + (1.0 - self.config.svdd_lambda) * recon,
            },
        )

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        if round_idx <= int(self.config.phase1_rounds):
            return self._phase1_result(client_state_dicts)
        return self._phase2_result(client_state_dicts)


__all__ = ["SVDDDefense", "_lower_quantile_mask"]
