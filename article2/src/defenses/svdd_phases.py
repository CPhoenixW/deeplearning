from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import DefenseResult as RoundStats


@dataclass(frozen=True)
class ReconstructionObjective:
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean(dim=1)


@dataclass(frozen=True)
class SVDDObjective:
    def __call__(self, embedding: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return ((embedding - center) ** 2).sum(dim=1)


class WarmupPhase:
    name = "warmup"

    def run(self, defense, round_idx: int, client_states) -> RoundStats:
        center_norm, z_var, ae_loss, scores, keep_mask = defense.phase1_step(
            round_idx, client_states
        )
        weights = keep_mask.float() / (keep_mask.float().sum() + 1e-12)
        kept = int(keep_mask.sum().item())
        total = len(client_states)
        return RoundStats(
            center_norm=center_norm,
            z_var=z_var,
            ae_loss=ae_loss,
            svdd_loss=float("nan"),
            d=scores,
            m=keep_mask.float(),
            alpha=weights,
            phase="warmup",
            show_detection=True,
            monitor_items=[
                ("Defense", "SVDD"),
                ("Phase-1 Selector", str(defense.config.phase1_selection)),
                ("Feature Mode", str(defense.config.svdd_feature_mode)),
                ("Kept clients", f"{kept}/{total}"),
                ("Center L2-Norm", f"{center_norm:.6f}"),
                ("Z-Space Variance", f"{z_var:.6f}"),
                ("AE L1-Loss", f"{ae_loss:.6f}"),
            ],
            recon_loss=ae_loss,
            total_loss=ae_loss,
            participant_metrics={"reconstruction_loss": scores.detach().cpu()},
        )


class CenterInitialization:
    name = "center_initialization"

    def ensure(self, defense, client_states) -> None:
        if defense.c is None:
            defense.init_center(client_states)


class FilteringPhase:
    name = "filtering"

    def __init__(self) -> None:
        self.center = CenterInitialization()

    def run(self, defense, round_idx: int, client_states) -> RoundStats:
        self.center.ensure(defense, client_states)
        svdd_round = max(1, round_idx - int(defense.config.phase1_rounds))
        (
            center_norm,
            z_var,
            svdd_loss,
            recon_loss,
            tau,
            threshold,
            scores,
            accepted,
            weights,
            recon_per_client,
        ) = defense.phase2_step(svdd_round, client_states)
        kept = int(accepted.sum().item())
        total = len(client_states)
        total_loss = (
            defense.config.svdd_loss_weight * svdd_loss
            + defense.config.recon_loss_weight * recon_loss
        )
        total_per_client = (
            defense.config.svdd_loss_weight * scores
            + defense.config.recon_loss_weight * recon_per_client
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
                ("Feature Mode", str(defense.config.svdd_feature_mode)),
                ("Kept clients", f"{kept}/{total}"),
                ("Tau", f"{tau:.6f}"),
                ("Threshold", f"{threshold:.6f}"),
                ("Center L2-Norm", f"{center_norm:.6f}"),
                ("Z-Space Variance", f"{z_var:.6f}"),
            ],
            participant_metrics={
                "svdd_loss": scores.detach().cpu(),
                "reconstruction_loss": recon_per_client.detach().cpu(),
                "total_loss": total_per_client.detach().cpu(),
            },
        )
