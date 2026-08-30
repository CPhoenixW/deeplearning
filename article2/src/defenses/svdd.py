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
    INPUT_DIM = 4096
    TOPK_REJECT_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50)

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
        self.phase1_score_mode = "recon"
        self.phase2_score_mode = "combined"
        self.validation_tie_break = str(
            getattr(config, "svdd_validation_tie_break", "largest") or "largest"
        ).lower().strip()
        if self.validation_tie_break not in {"largest", "smallest", "median"}:
            raise ValueError(
                "svdd_validation_tie_break must be one of ('largest', 'smallest', 'median')."
            )
        self.input_mode = "absolute"
        self.normalization_eps = float(config.svdd_normalization_eps)
        if not math.isfinite(self.normalization_eps) or self.normalization_eps <= 0.0:
            raise ValueError("svdd_normalization_eps must be positive and finite.")
        descriptor_device = str(getattr(config, "svdd_descriptor_device", "auto")).lower().strip()
        if descriptor_device == "auto":
            projection_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        else:
            projection_device = torch.device(descriptor_device)
        if projection_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("svdd_descriptor_device requests CUDA, but CUDA is unavailable.")
        reference = self.state_dict_for_clients()
        self._zero_reference = {
            name: torch.zeros_like(reference[name]) for name in self.param_names
        }
        self.descriptor = FixedHierarchicalMultiViewDescriptor(
            reference,
            parameter_names=self.param_names,
            output_dim=self.INPUT_DIM,
            seed=int(getattr(config, "svdd_descriptor_seed", 2027)),
            projection_device=projection_device,
            global_ratio=float(getattr(config, "svdd_descriptor_global_ratio", 0.5)),
            layer_ratio=float(getattr(config, "svdd_descriptor_layer_ratio", 0.375)),
            statistics_ratio=float(getattr(config, "svdd_descriptor_statistics_ratio", 0.125)),
        )
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
        best_accuracy = float("-inf")
        candidates_by_ratio: dict[float, tuple[Tensor, Tensor]] = {}
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
            candidates_by_ratio[reject_ratio] = (mask, weights)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        tied_ratios = [
            ratio
            for ratio in self.TOPK_REJECT_RATIOS
            if candidate_accuracies[f"{ratio:.2f}"] == best_accuracy
        ]
        if self.validation_tie_break == "median" and len(tied_ratios) > 1:
            ordered = sorted(tied_ratios)
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                best_ratio = ordered[midpoint]
            else:
                best_ratio = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
            best_mask, best_weights = self._topk_candidate(
                finite_order, best_ratio, len(client_state_dicts)
            )
        else:
            if self.validation_tie_break == "smallest":
                best_ratio = min(tied_ratios)
            else:
                best_ratio = max(tied_ratios)
            best_mask, best_weights = candidates_by_ratio[best_ratio]
        if best_mask is None or best_weights is None:
            raise RuntimeError("Top-K validation selection produced no candidate.")
        selected_state = weighted_fedavg(
            client_state_dicts,
            best_weights,
            device=self.aggregation_device,
        )
        self.global_model.load_state_dict(selected_state)
        return best_mask.float(), best_weights, best_ratio, best_accuracy, candidate_accuracies

    def _select_by_mad_threshold(
        self,
        scores: Tensor,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, float, float, Dict[str, float]]:
        """Select clients using the configured median-plus-k-MAD cutoff."""

        if scores.ndim != 1 or scores.numel() != len(client_state_dicts):
            raise ValueError("MAD scores must match the client count.")
        finite = torch.isfinite(scores)
        if not bool(finite.any().item()):
            raise FloatingPointError("All client scores are non-finite.")
        k = float(getattr(self.config, "svdd_mad_k", 3.0))
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("svdd_mad_k must be finite and non-negative.")
        valid_scores = scores[finite]
        median = valid_scores.median()
        mad = (valid_scores - median).abs().median()
        threshold = median + k * mad
        accepted = finite & (scores <= threshold)
        # The median itself is normally accepted; retain a deterministic
        # fallback for unusual floating-point edge cases.
        if not bool(accepted.any().item()):
            finite_indices = torch.where(finite)[0]
            accepted[finite_indices[torch.argmin(valid_scores)]] = True
        weights = accepted.float() / accepted.sum().clamp_min(1.0)
        selected_state = weighted_fedavg(
            client_state_dicts, weights, device=self.aggregation_device
        )
        self.global_model.load_state_dict(selected_state)
        reject_ratio = 1.0 - float(accepted.sum().item()) / float(len(client_state_dicts))
        return accepted.float(), weights, reject_ratio, float("nan"), {}

    def _select_clients(
        self,
        scores: Tensor,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, float, float, Dict[str, float]]:
        method = str(getattr(self.config, "svdd_selection_method", "topk_validation"))
        method = method.lower().strip()
        if method == "mad_threshold":
            return self._select_by_mad_threshold(scores, client_state_dicts)
        if method == "topk_validation":
            return self._select_topk_by_validation(scores, client_state_dicts)
        raise ValueError(
            "svdd_selection_method must be 'topk_validation' or 'mad_threshold'."
        )

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
        """Map absolute client weights into the fixed descriptor."""

        return self.descriptor.describe_many(client_state_dicts, self._zero_reference)

    def _scale_input_matrix(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply finite-client feature-wise median/MAD scaling."""

        finite_rows = torch.isfinite(raw).all(dim=1)
        if not bool(finite_rows.any().item()):
            raise FloatingPointError("All SVDD feature rows are non-finite.")
        safe = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = safe[finite_rows]
        center = valid.median(dim=0).values
        scale = (valid - center).abs().median(dim=0).values
        # Gaussian-consistent MAD scale; the final clamp handles constant
        # descriptor coordinates without introducing non-finite values.
        scale = scale * 1.4826
        scale = scale.clamp_min(self.normalization_eps)
        scaled = (safe - center) / scale
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

    def phase1_step(
        self,
        client_state_dicts: List[Dict[str, Tensor]],
        *,
        initialize_center: bool = False,
    ) -> tuple:
        """Run one warm-up round: detect first, then update AE.

        Phase 1 deliberately uses only reconstruction error for detection.  The
        AE is evaluated before the optimizer step, so the same-round client
        decision cannot be affected by the update it is about to receive.
        """

        raw_X = self._build_input_matrix(client_state_dicts)  # (K, D_feat)
        X, finite_rows = self._scale_input_matrix(raw_X)
        finite_rows_device = finite_rows.to(self.device)
        X_device = X.to(self.device)

        # Detect with the AE state from the beginning of this round.  No SVDD
        # distance or center is needed in Phase 1.
        self.ae.eval()
        with torch.no_grad():
            per_client_loss = self._reconstruction_loss(
                self.ae(X_device), X_device
            )
            per_client_loss = torch.where(
                finite_rows_device,
                per_client_loss,
                torch.full_like(per_client_loss, float("inf")),
            )
            selection_scores = per_client_loss

        keep_mask, weights, selected_ratio, validation_accuracy, candidates = (
            self._select_clients(
                selection_scores.detach().cpu(), client_state_dicts
            )
        )

        # Update the encoder and decoder only after client selection, and only
        # with the selected finite clients.
        trusted = keep_mask.to(self.device, dtype=torch.bool) & finite_rows_device
        if not bool(trusted.any().item()):
            raise FloatingPointError("No trusted finite clients are available for AE update.")
        self.ae.train()
        x_trusted = X_device[trusted]
        per_sample_loss = self._reconstruction_loss(self.ae(x_trusted), x_trusted)
        loss = per_sample_loss.mean()
        self._safe_ae_step(loss, self.config.ae_grad_clip)

        # The last Phase-1 round initializes c after the AE update.  Therefore
        # c and the AE used in the first Phase-2 round share the same latent
        # space, and Phase 2 can consume c without recalculating a new mask.
        self.ae.eval()
        with torch.no_grad():
            embeddings = self.ae.encode(X_device)
            if initialize_center:
                self._set_center_from_trusted_embeddings(embeddings, trusted)
            Z = embeddings
            finite_z = Z[torch.isfinite(Z).all(dim=1)]
            z_var = finite_z.var().item() if finite_z.numel() > 1 else 0.0
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

    def _set_center_from_trusted_embeddings(
        self,
        embeddings: Tensor,
        trusted_mask: Tensor,
    ) -> Tuple[float, float]:
        """Initialize c from an already selected trusted client set."""

        trusted = trusted_mask.to(self.device, dtype=torch.bool)
        finite = torch.isfinite(embeddings).all(dim=1)
        trusted = trusted & finite
        if not bool(trusted.any().item()):
            raise FloatingPointError(
                "No trusted finite clients are available to initialize SVDD center."
            )
        with torch.no_grad():
            c = embeddings[trusted].mean(dim=0)
            if not bool(torch.isfinite(c).all().item()):
                raise FloatingPointError("SVDD center initialization produced non-finite values.")
            c = c.detach().clone()
            c[c.abs() < 0.01] = 0.01
            self.c = c
            finite_z = embeddings[finite]
            z_var = finite_z.var().item() if finite_z.numel() > 1 else 0.0
        return float(self.c.norm().item()), float(z_var)

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
        recon_rank = self._rank_score(recon_per_client.to(self.device))
        svdd_rank = self._rank_score(d)
        selection_scores = 0.5 * (recon_rank + svdd_rank)
        valid_scores = torch.isfinite(recon_per_client.to(self.device)) & torch.isfinite(d)
        selection_scores = torch.where(
            valid_scores,
            selection_scores,
            torch.full_like(selection_scores, float("inf")),
        )
        accepted_mask, aggregation_weights, selected_ratio, validation_accuracy, candidates = (
            self._select_clients(
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

        # The reconstruction branch is also restricted to the clients that
        # passed the current-round detection.  The optional quantile is applied
        # only inside that trusted subset, so rejected clients never contribute
        # to the encoder/decoder update.
        trusted_feature_rows = trusted_cpu & finite_feature_rows
        if not bool(trusted_feature_rows.any().item()):
            raise FloatingPointError("No trusted finite clients are available for AE update.")
        X_cur = X[trusted_feature_rows].to(self.device)
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

    def _phase1_result(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        center, variance, loss, scores, recon, accepted, ratio, accuracy, candidates = (
            self.phase1_step(
                client_state_dicts,
                initialize_center=round_idx == int(self.config.phase1_rounds),
            )
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
            return self._phase1_result(round_idx, client_state_dicts)
        return self._phase2_result(client_state_dicts)


__all__ = ["SVDDDefense", "_lower_quantile_mask"]
