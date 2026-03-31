from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

try:
    from .config import FedConfig
    from .feature_buffer import BNReplayBuffer
    from .models import AutoEncoder
    from .utils import (
        aggregate_updates_with_info,
        build_bn_matrix,
        compute_multi_krum_scores,
        robust_scale_features,
        weighted_fedavg,
    )
except ImportError:
    from config import FedConfig
    from feature_buffer import BNReplayBuffer
    from models import AutoEncoder
    from utils import (
        aggregate_updates_with_info,
        build_bn_matrix,
        compute_multi_krum_scores,
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

    def state_dict_for_clients(self) -> Dict[str, Tensor]:
        sd = self.global_model.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        raise NotImplementedError


class FedAvgServer(BaseServer):
    defense_name = "fedavg"

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
            phase="fedavg Baseline",
            show_detection=True,
            monitor_items=[
                ("Defense", "FedAvg"),
                ("Clients Kept", f"{int(m.sum().item())}/{k}"),
                ("Uniform Weight", f"{(1.0 / max(1, k)):.6f}"),
            ],
        )


class TrimmedMeanServer(BaseServer):
    defense_name = "trimmed_mean"

    def aggregate(self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]) -> RoundStats:
        k = len(client_state_dicts)
        global_sd, m, alpha = aggregate_updates_with_info(
            client_state_dicts,
            method="trimmed_mean",
            trim_ratio=self.config.trimmed_mean_ratio,
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
            phase="trimmed_mean Baseline",
            show_detection=True,
            monitor_items=[
                ("Defense", "Trimmed Mean"),
                ("Trim Ratio", f"{self.config.trimmed_mean_ratio:.4f}"),
                ("Clients Kept", f"{int(m.sum().item())}/{k}"),
            ],
        )


class MultiKrumServer(BaseServer):
    defense_name = "multi_krum"

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
            phase="multi_krum Baseline",
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
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        # 限制潜在空间维度，避免高维距离退化
        latent_dim = min(config.latent_dim, 64)
        self.ae = AutoEncoder(d_bn=d_bn, latent_dim=latent_dim).to(self.device)
        self.buffer = BNReplayBuffer(capacity=config.buffer_capacity, d_bn=d_bn)

        self.c: Optional[Tensor] = None

        self.optimizer_ae = torch.optim.Adam(
            self.ae.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _state_dict_for_clients(self) -> Dict[str, Tensor]:
        """Return a detached CPU copy of global state_dict for broadcasting."""

        sd = self.global_model.state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    # --------------------------------------------------------------------- #
    # Phase 1: AE warm-up
    # --------------------------------------------------------------------- #
    def phase1_step(
        self, round_idx: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, Tensor, Tensor]:
        """Run one AE warm-up round and uniform FedAvg aggregation.

        Returns:
            center_norm, z_variance, ae_loss,
            per-client reconstruction loss on current round, keep_mask
        """

        # 当前轮所有客户端的 BN 特征（仅用于监控 per-client loss）
        X = build_bn_matrix(client_state_dicts)  # (K, D_bn)
        X = robust_scale_features(X)

        # 先在当前轮上做一次重构，得到每个客户端的 L1 损失并做 trimmed
        self.ae.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            x_hat_cur = self.ae(X_dev)
            per_client_loss = (x_hat_cur - X_dev).abs().sum(dim=1)  # (K,)
            q80_cur = torch.quantile(per_client_loss.detach(), 0.6)
            keep_mask = per_client_loss <= q80_cur

        # 将当前轮特征加入 buffer，再用 buffer 训练 AE（仍然是 trimmed loss）
        self.buffer.add(X)

        n_buf = len(self.buffer)
        batch_size = min(128, n_buf)
        X_train = self.buffer.sample(batch_size).to(self.device)

        self.ae.train()
        x_hat = self.ae(X_train)
        per_sample_loss = (x_hat - X_train).abs().sum(dim=1)
        q80 = torch.quantile(per_sample_loss.detach(), 0.8)
        keep = per_sample_loss <= q80
        loss = per_sample_loss[keep].mean()

        self.optimizer_ae.zero_grad()
        loss.backward()
        clip_grad_norm_(self.ae.parameters(), self.config.ae_grad_clip)
        self.optimizer_ae.step()

        # Uniform FedAvg
        K = len(client_state_dicts)
        alpha = torch.full((K,), 1.0 / K)
        global_sd = weighted_fedavg(client_state_dicts, alpha)
        self.global_model.load_state_dict(global_sd)

        with torch.no_grad():
            self.ae.eval()
            Z = self.ae.encode(X.to(self.device))
            z_var = Z.var().item()
            center_norm = 0.0 if self.c is None else float(self.c.norm().item())

        return center_norm, z_var, float(loss.item()), per_client_loss.detach().cpu(), keep_mask.detach().cpu()

    # --------------------------------------------------------------------- #
    # Transition: center initialization
    # --------------------------------------------------------------------- #
    def init_center(self, client_state_dicts: List[Dict[str, Tensor]]) -> Tuple[float, float]:
        """Initialize SVDD center c using well-reconstructed clients."""

        X = build_bn_matrix(client_state_dicts)
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

    # --------------------------------------------------------------------- #
    # Phase 2: SVDD filtering
    # --------------------------------------------------------------------- #
    def phase2_step(
        self, svdd_round: int, client_state_dicts: List[Dict[str, Tensor]]
    ) -> Tuple[float, float, float, float, Tensor, Tensor, Tensor]:
        """Run one SVDD-filtered aggregation round.

        Returns:
            center_norm, z_variance, svdd_loss_value, recon_loss_value,
            d, M, alpha
        """

        assert self.c is not None, "SVDD center c must be initialized before Phase 2."

        X = build_bn_matrix(client_state_dicts)
        X = robust_scale_features(X)
        self.buffer.add(X)

        # Embeddings without grad
        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X.to(self.device))

        # Distances to center
        c = self.c.to(self.device)
        d = ((Z - c) ** 2).sum(dim=1)  # (K,)

        # Robust threshold using MAD with annealed tau:
        # keep more clients early, tighten filtering later.
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

        # Hard mask only: uniform FedAvg over kept clients (no credit / no soft weights).
        alpha = M / (M.sum() + 1e-12)

        # Center EMA update using trusted clients
        trusted = M > 0.5
        if trusted.sum() > 0:
            with torch.no_grad():
                c_new = Z[trusted].mean(dim=0)
                c_updated = self.config.center_ema_decay * c + (1.0 - self.config.center_ema_decay) * c_new
                c_updated[c_updated.abs() < 0.01] = 0.01
                self.c = c_updated.detach()

        # SVDD fine-tune encoder with reconstruction anchor
        self.ae.train()
        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = False

        # mask 需在与 X 相同的设备上；这里 X 在 CPU 上
        trusted_cpu = trusted.detach().cpu()
        X_trusted = X[trusted_cpu]
        Z_trusted = self.ae.encode(X_trusted.to(self.device))
        svdd_loss = ((Z_trusted - self.c.detach().to(self.device)) ** 2).sum(dim=1).mean()

        for p_ in self.ae.decoder.parameters():
            p_.requires_grad = True

        # Reconstruction anchor from buffer (trimmed)
        X_buf = self.buffer.sample(min(64, len(self.buffer))).to(self.device)
        X_buf_hat = self.ae(X_buf)
        recon_per_sample = (X_buf_hat - X_buf).abs().mean(dim=1)
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
            return RoundStats(
                center_norm=center_norm,
                z_var=z_var,
                ae_loss=ae_loss,
                svdd_loss=float("nan"),
                d=d,
                m=keep_mask.float(),
                alpha=torch.full((len(client_state_dicts),), 1.0 / len(client_state_dicts)),
                phase="AE Warm-up",
                show_detection=True,
                monitor_items=[
                    ("Defense", "SVDD"),
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
            phase="SVDD Filtering",
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
    "fedavg": FedAvgServer,
    "trimmed_mean": TrimmedMeanServer,
    "multi_krum": MultiKrumServer,
    "svdd": SVDDServer,
}

# Backward compatibility
FederatedServer = SVDDServer

