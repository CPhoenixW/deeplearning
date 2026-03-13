# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - 模型定义
ResNet18（全局分类）、无偏置 Encoder、Decoder、两阶段 AE-SVDD 用 AutoEncoder。
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Optional


# 潜在空间维度（可配置；由 32 调整为 128）
LATENT_DIM = 128


def get_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """返回用于 CIFAR-10 的 ResNet18 实例（torchvision 标准结构，与 BN 提取兼容）。"""
    return resnet18(weights=None, num_classes=num_classes)


# --------------- 两阶段 AE-SVDD 检测网络 ---------------


class Encoder(nn.Module):
    """
    BN 特征 -> 潜在空间（默认 128 维）。
    【致命约束】所有 Linear 层 bias=False，最后一层无激活。
    """

    def __init__(self, d_bn: int, d_out: int = LATENT_DIM):
        super().__init__()
        self.d_bn = d_bn
        self.d_out = d_out
        self.net = nn.Sequential(
            nn.Linear(d_bn, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_out, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """潜在向量 -> 重建 D_bn 维。Linear 层可有 bias。"""

    def __init__(self, d_out: int, d_bn: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_out, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, d_bn),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoEncoder(nn.Module):
    """
    两阶段 AE-SVDD 用自编码器。
    - forward(x): 重建结果，形状与 x 一致。
    - encode(x): 仅潜在特征，形状 (..., 32)。
    """

    def __init__(self, d_bn: int, d_out: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(d_bn=d_bn, d_out=d_out)
        self.decoder = Decoder(d_out=d_out, d_bn=d_bn)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
