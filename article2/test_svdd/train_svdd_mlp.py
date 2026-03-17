from __future__ import annotations

"""
中心化场景下，用三层全连接 AutoEncoder 预训练 + SVDD 在 BN 特征上做异常检测。

数据结构假定为 prepare_cifar10_subset.py 生成的：
    {
        "features": Tensor [N, D_bn],
        "labels":   Tensor [N] 或 None,
        "is_malicious": Tensor [N], 0=benign,1=malicious（仅用于评估，不参与训练/阈值）
    }

用法示例（在 article2 目录下）：
    python -m test_svdd.train_svdd_mlp --data cifar10.pth
"""

import argparse

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from new.config import FedConfig


class ThreeLayerEncoder(nn.Module):
    """三层全连接编码器：D_bn -> 1024 -> 512 -> latent_dim."""

    def __init__(self, d_bn: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_bn, 1024, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 512, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, latent_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def mad_1d(x: Tensor) -> Tensor:
    med = x.median()
    return 1.4826 * (x - med).abs().median()


class ThreeLayerDecoder(nn.Module):
    """Encoder 的对称解码器：latent_dim -> 512 -> 1024 -> D_bn。"""

    def __init__(self, latent_dim: int, d_bn: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 1024, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, d_bn, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10.pth", help="中心化 BN 特征数据文件")
    parser.add_argument("--ae_epochs", type=int, default=200, help="AutoEncoder 预训练轮数")
    parser.add_argument("--epochs", type=int, default=200, help="SVDD 训练轮数")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    cfg = FedConfig()
    device = torch.device("cuda" if (cfg.device in ["cuda", "auto"] and torch.cuda.is_available()) else "cpu")

    # 加载数据
    data = torch.load(args.data, map_location="cpu")
    X: Tensor = data["features"].float()
    is_mal: Tensor = data.get("is_malicious", torch.zeros(X.size(0), dtype=torch.long))
    N, d_bn = X.shape
    print(f"[INFO] 载入中心化数据: N={N}, D_bn={d_bn}, 恶意标记={is_mal.sum().item()} 条")

    # 特征级鲁棒标准化：按维度用 median + MAD 归一化，以抵抗 20% 噪声
    X_med = X.median(dim=0).values
    X_mad = (X - X_med).abs().median(dim=0).values
    X_mad[X_mad < 1e-4] = 1e-4
    X = (X - X_med) / X_mad

    # 三层全连接 AutoEncoder （强制限制 latent 维度，避免维度灾难）
    latent_dim = min(cfg.latent_dim, 64)
    encoder = ThreeLayerEncoder(d_bn=d_bn, latent_dim=latent_dim).to(device)
    decoder = ThreeLayerDecoder(latent_dim=latent_dim, d_bn=d_bn).to(device)
    opt_ae = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.ae_lr,
        weight_decay=cfg.ae_weight_decay,
    )

    X_dev = X.to(device)

    # ---- Phase 1: AutoEncoder 预训练（Trimmed MSE 重构）----
    mse_loss_fn = nn.MSELoss(reduction="none")
    for epoch in range(1, args.ae_epochs + 1):
        perm = torch.randperm(N)
        X_shuf = X_dev[perm]

        encoder.train()
        decoder.train()
        total_recon = 0.0
        total_cnt = 0

        for i in range(0, N, args.batch_size):
            xb = X_shuf[i : i + args.batch_size]
            z = encoder(xb)
            x_hat = decoder(z)
            # 每个样本的重构误差，使用 trimmed loss 只保留前 80% 最小误差
            raw_loss = mse_loss_fn(x_hat, xb).mean(dim=1)
            keep_ratio = 0.80
            k_ae = max(1, int(xb.size(0) * keep_ratio))
            loss = raw_loss.topk(k_ae, largest=False).values.mean()

            opt_ae.zero_grad()
            loss.backward()
            clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), cfg.ae_grad_clip)
            opt_ae.step()

            total_recon += loss.item() * xb.size(0)
            total_cnt += xb.size(0)

        with torch.no_grad():
            encoder.eval()
            Z_full = encoder(X_dev)
            z_var = Z_full.var().item()
        print(
            f"[AE-MLP] Epoch {epoch:03d}  recon={total_recon / max(1, total_cnt):.4f}  z_var={z_var:.4f}"
        )

    # ---- 初始化中心 c：用预训练好的 encoder 在全体样本上的 embedding 中位数 ----
    encoder.eval()
    with torch.no_grad():
        Z_init = encoder(X_dev)
        c = Z_init.median(dim=0).values
        c[c.abs() < 0.01] = 0.01
    print(f"[SVDD-MLP] init center norm = {c.norm().item():.4f}")

    # SVDD 阶段仅优化 encoder
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay)

    # ---- 训练：最小化到中心的距离（纯 SVDD，无重构项）----
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N)
        X_shuf = X_dev[perm]

        encoder.train()
        total_svdd = 0.0
        total_cnt = 0

        for i in range(0, N, args.batch_size):
            xb = X_shuf[i : i + args.batch_size]

            z = encoder(xb)
            # 截断 SVDD 损失：仅对距离中心最近的 80% 样本反向传播
            dists = ((z - c.to(device)) ** 2).sum(dim=1)
            keep_ratio = 0.80
            k = max(1, int(xb.size(0) * keep_ratio))
            svdd_loss = dists.topk(k, largest=False).values.mean()

            opt.zero_grad()
            svdd_loss.backward()
            clip_grad_norm_(encoder.parameters(), cfg.svdd_grad_clip)
            opt.step()

            total_svdd += svdd_loss.item() * xb.size(0)
            total_cnt += xb.size(0)

        # 监控：全量距离 + MAD 阈值 + TPR/FPR
        encoder.eval()
        with torch.no_grad():
            Z_all = encoder(X_dev)
            d = ((Z_all - c.to(device)) ** 2).sum(dim=1).cpu()
            med_d = d.median()
            mad_d = mad_1d(d)
            mad_d = max(mad_d.item(), 1e-6)
            thr = med_d.item() + cfg.tau_multiplier * mad_d
            M = (d <= thr).long()

            benign_mask = (is_mal == 0)
            mal_mask = (is_mal == 1)
            rej = (M == 0)

            if mal_mask.any():
                tpr = float(rej[mal_mask].float().mean().item())
            else:
                tpr = 0.0
            if benign_mask.any():
                fpr = float(rej[benign_mask].float().mean().item())
            else:
                fpr = 0.0

            center_norm = c.norm().item()
            d_ben = d[benign_mask]
            d_mal = d[mal_mask]
            d_ben_mean = float(d_ben.mean().item()) if d_ben.numel() > 0 else 0.0
            d_mal_mean = float(d_mal.mean().item()) if d_mal.numel() > 0 else 0.0
            d_ben_med = float(d_ben.median().item()) if d_ben.numel() > 0 else 0.0
            d_mal_med = float(d_mal.median().item()) if d_mal.numel() > 0 else 0.0

        print(
            f"[SVDD-MLP] Epoch {epoch:03d}  "
            f"svdd={total_svdd/max(1,total_cnt):.4f}  "
            f"center_norm={center_norm:.4f}  "
            f"d_ben(mean/med)={d_ben_mean:.4f}/{d_ben_med:.4f}  "
            f"d_mal(mean/med)={d_mal_mean:.4f}/{d_mal_med:.4f}  "
            f"TPR={tpr:.4f}  FPR={fpr:.4f}"
        )

    # ---- 训练结束后，逐客户端输出诊断结果 ----
    print("\n[SVDD-MLP] Per-sample detailed results:")
    for i in range(N):
        dist_i = d[i].item()
        is_rejected = bool(rej[i].item())
        is_malicious = bool(is_mal[i].item())
        pred_str = "Malicious" if is_rejected else "Benign"
        truth_str = "Malicious" if is_malicious else "Benign"
        print(
            f"[Result] Client {i:03d} | Dist: {dist_i:.4f} | "
            f"Threshold: {thr:.4f} | Pred: {pred_str} | Truth: {truth_str}"
        )


if __name__ == "__main__":
    main()

