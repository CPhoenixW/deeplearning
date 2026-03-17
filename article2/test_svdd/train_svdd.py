from __future__ import annotations

"""
在中心化场景下，用 cifar10.pth 中的 BN 特征训练 AE + SVDD。

数据结构假定为 prepare_cifar10_subset.py 生成的：
    {
        "features": Tensor [N, D_bn],
        "labels":   Tensor [N] 或 None,
        "is_malicious": Tensor [N], 0=benign,1=malicious（仅用于评估，不参与训练/阈值）
    }

用法示例（在 article2 目录下）：
    python -m test_svdd.train_svdd --data cifar10.pth
"""

import argparse

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from new.config import FedConfig
from new.models import AutoEncoder


def mad_1d(x: Tensor) -> Tensor:
    med = x.median()
    return 1.4826 * (x - med).abs().median()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10.pth", help="中心化 BN 特征数据文件")
    parser.add_argument("--epochs_ae", type=int, default=200, help="AE 纯重构训练轮数")
    parser.add_argument("--epochs_svdd", type=int, default=200, help="SVDD 阶段训练轮数")
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

    # 构建 AE
    ae = AutoEncoder(d_bn=d_bn, latent_dim=cfg.latent_dim).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay)

    # -------- Phase 1: 仅 AE 重构训练 --------
    X_dev = X.to(device)
    for epoch in range(1, args.epochs_ae + 1):
        perm = torch.randperm(N)
        X_shuf = X_dev[perm]
        total_loss = 0.0
        total_cnt = 0

        ae.train()
        for i in range(0, N, args.batch_size):
            xb = X_shuf[i : i + args.batch_size]
            x_hat = ae(xb)
            per_sample = (x_hat - xb).abs().sum(dim=1)
            q80 = torch.quantile(per_sample.detach(), 0.8)
            keep = per_sample <= q80
            loss = per_sample[keep].mean()

            opt_ae.zero_grad()
            loss.backward()
            clip_grad_norm_(ae.parameters(), cfg.ae_grad_clip)
            opt_ae.step()

            total_loss += loss.item() * keep.sum().item()
            total_cnt += keep.sum().item()

        with torch.no_grad():
            Z_full = ae.encode(X_dev)
            z_var = Z_full.var().item()
        print(f"[AE] Epoch {epoch:03d}  loss={total_loss/max(1,total_cnt):.4f}  z_var={z_var:.4f}")

    # -------- 初始化中心 c（用重构误差较小的一半样本）--------
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(X_dev)
        X_hat = ae(X_dev)
        recon = (X_hat - X_dev).abs().sum(dim=1)
        med = recon.median()
        init_mask = recon <= med
        c = Z[init_mask].mean(dim=0)
        c[c.abs() < 0.01] = 0.01
    print(f"[SVDD] init center norm = {c.norm().item():.4f}")

    # -------- Phase 2: SVDD 训练（带重构锚点）--------
    # 只更新 encoder，decoder 用于重构约束
    for p in ae.decoder.parameters():
        p.requires_grad = True
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay)

    for epoch in range(1, args.epochs_svdd + 1):
        perm = torch.randperm(N)
        X_shuf = X_dev[perm]

        ae.train()
        total_svdd = 0.0
        total_recon = 0.0
        total_cnt = 0

        for i in range(0, N, args.batch_size):
            xb = X_shuf[i : i + args.batch_size]

            # SVDD 部分
            z = ae.encode(xb)
            svdd_loss = ((z - c.to(device)) ** 2).sum(dim=1).mean()

            # 重构锚点 + trimmed
            x_hat = ae(xb)
            recon_per = (x_hat - xb).abs().sum(dim=1)
            q80 = torch.quantile(recon_per.detach(), 0.8)
            keep = recon_per <= q80
            recon_loss = recon_per[keep].mean()

            loss = svdd_loss + cfg.svdd_recon_lambda * recon_loss

            opt_ae.zero_grad()
            loss.backward()
            clip_grad_norm_(ae.parameters(), cfg.svdd_grad_clip)
            opt_ae.step()

            total_svdd += svdd_loss.item() * xb.size(0)
            total_recon += recon_loss.item() * keep.sum().item()
            total_cnt += xb.size(0)

        # 监控：用 MAD + tau_multiplier 计算阈值并评估 TPR/FPR
        ae.eval()
        with torch.no_grad():
            Z_all = ae.encode(X_dev)
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

            # 一些额外的诊断指标：中心范数、距离统计、benign/malicious 距离均值
            center_norm = c.norm().item()
            d_ben = d[benign_mask]
            d_mal = d[mal_mask]
            d_ben_mean = float(d_ben.mean().item()) if d_ben.numel() > 0 else 0.0
            d_mal_mean = float(d_mal.mean().item()) if d_mal.numel() > 0 else 0.0
            d_ben_med = float(d_ben.median().item()) if d_ben.numel() > 0 else 0.0
            d_mal_med = float(d_mal.median().item()) if d_mal.numel() > 0 else 0.0

        print(
            f"[SVDD] Epoch {epoch:03d}  "
            f"svdd={total_svdd/max(1,total_cnt):.4f}  "
            f"recon={total_recon/max(1,total_cnt):.4f}  "
            f"center_norm={center_norm:.4f}  "
            f"d_ben(mean/med)={d_ben_mean:.4f}/{d_ben_med:.4f}  "
            f"d_mal(mean/med)={d_mal_mean:.4f}/{d_mal_med:.4f}  "
            f"TPR={tpr:.4f}  FPR={fpr:.4f}"
        )


if __name__ == "__main__":
    main()

