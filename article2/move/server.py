# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - Server 端两阶段 AE-SVDD 聚合与监控
步骤 A：BN 特征矩阵；Phase 1：AE 预训练 + FedAvg（均匀权重）；Phase 2：SVDD 过滤 + 软权重 FedAvg。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional

from utils import build_bn_feature_matrix, get_bn_feature_dim
from models import get_resnet18_cifar10, AutoEncoder, LATENT_DIM


# 阶段划分与 SVDD 退火相关超参数
PHASE1_END_ROUND = 50  # Round 1~20: AE 预训练；21~: SVDD 阶段

SVDD_WARMUP_ROUNDS = 50  # SVDD 退火窗口期（前 10 个 SVDD 轮内从软到硬）

# 温度 T：从极其平缓到极其陡峭
T_START = 5.0
T_END = 0.5

# 边界系数 k：从极度包容到极度严格
K_START = 3.0
K_END = 1.0

# SVDD Loss 权重 λ：从微调到强拉力
LAMBDA_START = 0.1
LAMBDA_END = 1.0


class FederatedServer:
    """两阶段 AE-SVDD 联邦聚合服务器。"""

    def __init__(
        self,
        device: torch.device,
        lr_ae: float = 1e-3,
        test_loader: Optional[DataLoader] = None,
    ):
        self.device = device
        self.test_loader = test_loader
        self.lr_ae = lr_ae
        # 是否使用球心的动量更新（True=动量；False=完全跟随当前纯净集中心）
        self.use_center_momentum: bool = True

        self.global_model = get_resnet18_cifar10(num_classes=10).to(device)
        self._d_bn = get_bn_feature_dim(self.global_model.state_dict())
        self.ae = AutoEncoder(d_bn=self._d_bn, d_out=LATENT_DIM).to(device)
        self.optimizer_ae = torch.optim.Adam(self.ae.parameters(), lr=lr_ae)

        self.c: Optional[torch.Tensor] = None  # 球心，Phase 2 使用
        # 良性客户端掩码（仅良性参与方可更新 SVDD 模型），由外部 ground_truth 设置
        self._benign_mask: Optional[torch.Tensor] = None

        # 每轮聚合后写入，供 evaluate_system_state 使用
        self._last_round = 0
        self._last_phase: str = ""
        self._last_X: Optional[torch.Tensor] = None
        self._last_Z: Optional[torch.Tensor] = None
        self._last_c: Optional[torch.Tensor] = None
        self._last_d: Optional[torch.Tensor] = None
        self._last_M: Optional[torch.Tensor] = None
        self._last_alpha: Optional[torch.Tensor] = None
        self._last_ae_loss: Optional[float] = None
        self._last_svdd_loss: Optional[float] = None
        # 记录 SVDD 退火参数，便于监控
        self._last_T: Optional[float] = None
        self._last_k: Optional[float] = None
        self._last_lambda: Optional[float] = None

    def set_ground_truth_labels(self, labels: List[int]) -> None:
        """
        设置全局的 ground-truth 标签（1=良性, 0=恶意）。
        只有良性客户端的参数可以参与 SVDD 模型（Encoder + 球心）的更新。
        """
        mask = torch.tensor(labels, dtype=torch.bool, device=self.device)
        self._benign_mask = mask

    def aggregate(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        round_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        执行步骤 A + 两阶段分流聚合，返回新一轮全局 ResNet18 state_dict。
        """
        K = len(client_state_dicts)
        device = self.device

        # ----- 步骤 A: 提取 BN 特征矩阵 X in R^{K x D_bn} -----
        X = build_bn_feature_matrix(client_state_dicts).float().to(device)
        self._last_round = round_num
        self._last_X = X.detach()
        if round_num <= PHASE1_END_ROUND:
            # ----- Phase 1: AE 预训练 -----
            self._last_phase = "AE Warm-up"
            self.ae.train()
            self.optimizer_ae.zero_grad()
            hat_X = self.ae(X)
            ae_loss = nn.functional.l1_loss(hat_X, X)
            ae_loss.backward()
            self.optimizer_ae.step()

            self._last_ae_loss = ae_loss.item()
            self._last_svdd_loss = None
            self._last_Z = self.ae.encode(X).detach()
            self._last_c = None
            self._last_d = None
            self._last_M = None
            self._last_alpha = None

            # Phase 1 聚合：均匀权重 FedAvg
            uniform_weights = torch.ones(K, device=device) / K
            global_sd = self._weighted_aggregate(client_state_dicts, uniform_weights.cpu())
        else:
            # ----- Phase 2: SVDD 阶段（硬截断 + 纯净集聚合 + 动态动量） -----
            self._last_phase = "SVDD Filtering"

            # 1) 提取当前轮的嵌入 Z
            self.ae.eval()
            with torch.no_grad():
                Z = self.ae.encode(X)
            self._last_Z = Z.detach()
            Z = Z.detach()

            # 1. 动态参数设置：p, k, lambda_svdd, beta
            svdd_round = round_num - PHASE1_END_ROUND
            warmup_rounds = SVDD_WARMUP_ROUNDS
            p = min(1.0, float(svdd_round) / float(warmup_rounds))

            k = K_START - p * (K_START - K_END)
            lambda_svdd = LAMBDA_START + p * (LAMBDA_END - LAMBDA_START)
            beta = 0.9 * p  # 动态动量：0 -> 0.9

            # 监控用：沿用 _last_T 槽位记录 beta，便于现有表格复用
            self._last_T = float(beta)
            self._last_k = float(k)
            self._last_lambda = float(lambda_svdd)

            # 2. 距离计算与硬截断
            if self.c is None:
                self.c = torch.median(Z, dim=0).values.detach().clone()
            d = torch.sum((Z - self.c) ** 2, dim=1)
            self._last_d = d.detach()

            median_d = torch.median(d).item()
            mad_d = torch.median(torch.abs(d - median_d)).item()
            mad_d = max(mad_d, 1e-4)
            tau = median_d + k * mad_d

            M = (d <= tau).float()
            if M.sum().item() < 0.5:
                M = torch.ones_like(M)
            self._last_M = M.detach()

            # 3. 纯净集权重与中心点更新
            # 硬权重：存活节点 (M=1) 平分权重，被截断节点严格 0
            alpha = M / (M.sum() + 1e-12)
            self._last_alpha = alpha.detach()

            # 纯净集中心点：只在 M=1 的客户端上计算中位数
            trusted_mask = M > 0.5
            Z_trusted = Z[trusted_mask]
            if Z_trusted.size(0) == 0:
                # 理论上不会发生（上面已经兜底），但为安全再保护一次
                Z_trusted = Z
            c_current = torch.median(Z_trusted, dim=0).values.detach()

            if self.c is None or not self.use_center_momentum:
                self.c = c_current.clone()
            else:
                self.c = (beta * self.c + (1.0 - beta) * c_current).detach()
            self._last_c = self.c.detach()

            # 4. Encoder SVDD 微调（仅根据纯净集掩码 M 施加约束）
            self.ae.train()
            self.optimizer_ae.zero_grad()
            for p_param in self.ae.decoder.parameters():
                p_param.requires_grad = False

            Z_enc = self.ae.encode(X)
            d_enc = torch.sum((Z_enc - self.c) ** 2, dim=1)

            svdd_mask = M
            svdd_norm = svdd_mask.sum() + 1e-12
            svdd_loss = lambda_svdd * (svdd_mask * d_enc).sum() / svdd_norm
            svdd_loss.backward()
            self.optimizer_ae.step()

            for p_param in self.ae.decoder.parameters():
                p_param.requires_grad = True

            self._last_ae_loss = None
            self._last_svdd_loss = svdd_loss.item()

            # 最后：使用刚算出的硬权重 alpha 进行全局聚合（Clean-set Aggregation）
            global_sd = self._weighted_aggregate(client_state_dicts, alpha.cpu())

        self.global_model.load_state_dict(global_sd, strict=True)
        return global_sd

    def _median_aggregate(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """同名参数在 client 维度取中位数。"""
        stacked = {}
        for key in client_state_dicts[0]:
            tensors = [sd[key] for sd in client_state_dicts]
            stacked[key] = torch.stack(tensors, dim=0)
        return {k: torch.median(v, dim=0).values for k, v in stacked.items()}

    def _weighted_aggregate(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """FedAvg：W = sum_k alpha_k W_k，weights 形状 (K,)。"""
        w = weights.to(self.device)
        out = {}
        for key in client_state_dicts[0]:
            tensors = [sd[key].to(self.device) for sd in client_state_dicts]
            stacked = torch.stack(tensors, dim=0)
            out[key] = (w.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0).cpu()
        return out

    def aggregate_fedavg(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        round_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        纯 FedAvg 聚合接口（对照实验用）：
        - 不使用 AE / SVDD，只做均匀加权的 FedAvg。
        - 仍然会更新 self.global_model，方便后续直接评估 test accuracy。
        """
        K = len(client_state_dicts)
        uniform_weights = torch.ones(K, device=self.device) / K
        global_sd = self._weighted_aggregate(client_state_dicts, uniform_weights.cpu())

        # 标记当前轮次与阶段，方便 evaluate_system_state 输出中区分
        self._last_round = round_num
        self._last_phase = "FedAvg-Baseline"
        self._last_X = None
        self._last_Z = None
        self._last_c = None
        self._last_d = None
        self._last_M = None
        self._last_alpha = None
        self._last_ae_loss = None
        self._last_svdd_loss = None
        self._last_T = None
        self._last_k = None
        self._last_lambda = None

        self.global_model.load_state_dict(global_sd, strict=True)
        return global_sd

    def _fmt(self, x: float, w: int = 10) -> str:
        """统一数值格式、固定宽度便于对齐。"""
        if x != x:
            return "N/A".rjust(w)
        return f"{x:.4f}".rjust(w)

    def evaluate_system_state(
        self,
        ground_truth_labels: List[int],
        client_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        每轮末尾调用，借鉴 nvidia-smi：上方为基本设置信息，下方为表格化参数输出。
        ground_truth_labels: 长度 K，1=良性，0=恶意；仅用于监控，不参与训练与权重计算。
        client_states: 可选，本轮各客户端上传的 state_dict，用于输出良性/恶意组参数前 5 维对比。
        """
        K = len(ground_truth_labels)
        benign_mask = torch.tensor(ground_truth_labels, dtype=torch.bool)
        malicious_mask = ~benign_mask
        num_benign = int(benign_mask.sum().item())
        num_malicious = K - num_benign

        # ----- 全局与 SVDD 状态量 -----
        center_norm = self._last_c.norm().item() if self._last_c is not None else float("nan")
        z_var = self._last_Z.var().item() if self._last_Z is not None else float("nan")
        ae_loss = self._last_ae_loss
        svdd_loss = self._last_svdd_loss

        # 全局测试集精度
        acc = float("nan")
        correct = total = 0
        if self.test_loader is not None:
            self.global_model.eval()
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    pred = self.global_model(images).argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total if total else 0.0

        width = 79
        border = "+" + "-" * (width - 2) + "+"
        lines: List[str] = []

        # ===== 顶部：基本设置（类似 nvidia-smi 顶部信息栏） =====
        lines.append(border)
        title = f" AE-SVDD Monitor "
        meta = f"Round {self._last_round}  Phase {self._last_phase}"
        header = (title + meta).ljust(width - 2)
        lines.append(f"|{header}|")

        settings = f"Clients {K}  Benign {num_benign}  Malicious {num_malicious}"
        settings = settings.ljust(width - 2)
        lines.append(f"|{settings}|")
        lines.append(border)

        # ===== Summary 表：核心标量指标 =====
        def summary_row(name: str, value: str) -> str:
            return f"| {name.ljust(24)} | {value.ljust(width - 2 - 3 - 24)}|"

        lines.append(summary_row("Center L2-Norm", f"{center_norm:.6f}" if center_norm == center_norm else "N/A"))
        lines.append(summary_row("Z-Space Variance", f"{z_var:.6f}" if z_var == z_var else "N/A"))
        if ae_loss is not None:
            lines.append(summary_row("AE L1-Loss", f"{ae_loss:.6f}"))
        if svdd_loss is not None:
            lines.append(summary_row("SVDD Loss", f"{svdd_loss:.6f}"))
        # 退火参数监控：[Annealing Params] 展示 T, k, lambda
        if self._last_T is not None and self._last_k is not None and self._last_lambda is not None:
            anneal_str = f"T={self._last_T:.2f}, k={self._last_k:.2f}, lambda={self._last_lambda:.2f}"
            lines.append(summary_row("Annealing Params", anneal_str))
        if acc == acc:
            acc_str = f"{acc:.4f}"
            if total > 0:
                acc_str += f"  ({correct}/{total})"
            lines.append(summary_row("Test Accuracy", acc_str))
        lines.append(border)

        # 默认情况下（Phase 1）TPR/FPR 不定义，用于日志时返回 None
        tpr: Optional[float] = None
        fpr: Optional[float] = None

        # ===== Detection 表：Benign vs Malicious（Phase 2 专属） =====
        if self._last_phase == "SVDD Filtering" and self._last_d is not None and self._last_alpha is not None:
            d = self._last_d.cpu()
            alpha = self._last_alpha.cpu()
            M = self._last_M.cpu()

            # 聚合级别统计（均值 + 权重总和）
            dist_benign = d[benign_mask].mean().item() if num_benign > 0 else float("nan")
            dist_malicious = d[malicious_mask].mean().item() if num_malicious > 0 else float("nan")
            weight_benign = alpha[benign_mask].sum().item() if num_benign > 0 else 0.0
            weight_malicious = alpha[malicious_mask].sum().item() if num_malicious > 0 else 0.0

            tpr = (M[malicious_mask] < 0.5).sum().item() / num_malicious if num_malicious > 0 else float("nan")
            fpr = (M[benign_mask] < 0.5).sum().item() / num_benign if num_benign > 0 else float("nan")

            # ---- 汇总表：Benign / Malicious 两类的整体统计 ----
            col1, col2, col3 = 22, 12, 12
            det_border = "+" + "-" * (col1 + 2) + "+" + "-" * (col2 + 2) + "+" + "-" * (col3 + 2) + "+"

            def det_row(c1: str, c2: str, c3: str) -> str:
                return (
                    "| "
                    + c1.ljust(col1)
                    + " | "
                    + c2.rjust(col2)
                    + " | "
                    + c3.rjust(col3)
                    + " |"
                )

            lines.append(det_border)
            lines.append(det_row("Detection (Phase 2)", "Benign", "Malicious"))
            lines.append(det_border)
            lines.append(
                det_row(
                    "Dist (avg)",
                    f"{dist_benign:.6f}" if dist_benign == dist_benign else "N/A",
                    f"{dist_malicious:.6f}" if dist_malicious == dist_malicious else "N/A",
                )
            )
            weight_avg_benign = weight_benign / num_benign if num_benign > 0 else float("nan")
            weight_avg_malicious = weight_malicious / num_malicious if num_malicious > 0 else float("nan")
            lines.append(
                det_row(
                    "Weight (avg)",
                    f"{weight_avg_benign:.6f}" if weight_avg_benign == weight_avg_benign else "N/A",
                    f"{weight_avg_malicious:.6f}" if weight_avg_malicious == weight_avg_malicious else "N/A",
                )
            )
            lines.append(det_row("TPR (mal. reject)", "-", f"{tpr:.4f}" if tpr == tpr else "N/A"))
            lines.append(det_row("FPR (ben. reject)", f"{fpr:.4f}" if fpr == fpr else "N/A", "-"))
            lines.append(det_border)

            # ---- 逐客户端明细表：每个参与方的距离 / 权重 / 截断标记 ----
            cid_w, type_w, dist_w, alpha_w, mask_w = 5, 8, 14, 14, 5
            row_border = (
                "+"
                + "-" * (cid_w + 2)
                + "+"
                + "-" * (type_w + 2)
                + "+"
                + "-" * (dist_w + 2)
                + "+"
                + "-" * (alpha_w + 2)
                + "+"
                + "-" * (mask_w + 2)
                + "+"
            )

            def row(c_id: int, is_benign: bool, dist_val: float, alpha_val: float, m_val: float) -> str:
                typ = "Benign" if is_benign else "Mal"
                return (
                    "| "
                    + str(c_id).rjust(cid_w)
                    + " | "
                    + typ.ljust(type_w)
                    + " | "
                    + (f"{dist_val:.6f}" if dist_val == dist_val else "N/A").rjust(dist_w)
                    + " | "
                    + f"{alpha_val:.6f}".rjust(alpha_w)
                    + " | "
                    + str(int(m_val)).rjust(mask_w)
                    + " |"
                )

            lines.append(row_border)
            lines.append(
                "| "
                + "ID".rjust(cid_w)
                + " | "
                + "Type".ljust(type_w)
                + " | "
                + "Dist".rjust(dist_w)
                + " | "
                + "Alpha".rjust(alpha_w)
                + " | "
                + "M".rjust(mask_w)
                + " |"
            )
            lines.append(row_border)
            for idx in range(K):
                lines.append(
                    row(
                        idx,
                        bool(benign_mask[idx].item()),
                        float(d[idx].item()),
                        float(alpha[idx].item()),
                        float(M[idx].item()),
                    )
                )
            lines.append(row_border)

        print("\n".join(lines))

        # 返回关键指标，供外部记录日志使用
        metrics: Dict[str, Optional[float]] = {
            "round": float(self._last_round),
            "center_norm": center_norm if center_norm == center_norm else None,
            "z_var": z_var if z_var == z_var else None,
            "ae_loss": ae_loss,
            "svdd_loss": svdd_loss,
            "test_acc": acc if acc == acc else None,
            "tpr": tpr if tpr is not None and tpr == tpr else None,
            "fpr": fpr if fpr is not None and fpr == fpr else None,
        }
        return metrics
