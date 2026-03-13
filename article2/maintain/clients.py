# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - 客户端实现
BenignClient（正常 SGD）、GaussianNoiseClient（高斯投毒）、
LabelFlippingClient（标签翻转）、SignFlippingClient（反向梯度/更新）。
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from models import get_resnet18_cifar10


# --------------- 基类 ---------------


class BaseClient:
    """客户端基类：接收全局 state_dict，返回本地 state_dict（及可选元信息）。"""

    def __init__(self, client_id: int, device: torch.device):
        self.client_id = client_id
        self.device = device

    def local_step(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        执行本地一步，返回要上传的模型 state_dict。
        """
        raise NotImplementedError


# --------------- 良性客户端 ---------------


class BenignClient(BaseClient):
    """良性客户端：使用下发的全局模型，在本地数据上执行 SGD 训练。"""

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 0.1,
        local_epochs: int = 1,
    ):
        super().__init__(client_id, device)
        self.dataloader = dataloader
        self.lr = lr
        self.local_epochs = local_epochs

    def local_step(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        model = get_resnet18_cifar10(num_classes=10).to(self.device)
        model.load_state_dict(global_state_dict, strict=True)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# --------------- 高斯投毒客户端 ---------------


class GaussianNoiseClient(BaseClient):
    """高斯投毒：不训练，直接对下发的全局模型参数加 N(0, sigma^2) 噪声后返回。"""

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        sigma: float = 1.0,
        lr: float = 0.1,
    ):
        super().__init__(client_id, device)
        self.sigma = sigma
        self.lr = lr
    def local_step(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        noisy_state = {}
        for k, v in global_state_dict.items():
            if v.is_floating_point():
                noise = torch.randn_like(v, device=v.device, dtype=v.dtype) * self.sigma
                noisy_state[k] = (v + noise).cpu().clone()
            else:
                noisy_state[k] = v.cpu().clone()
        return noisy_state


# --------------- 标签翻转客户端 ---------------


class LabelFlippingClient(BaseClient):
    """标签翻转：训练前对标签执行 y = 9 - y，再正常 SGD。本次主测试不实例化。"""

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 0.1,
        local_epochs: int = 1,
    ):
        super().__init__(client_id, device)
        self.dataloader = dataloader
        self.lr = lr
        self.local_epochs = local_epochs

    def local_step(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        model = get_resnet18_cifar10(num_classes=10).to(self.device)
        model.load_state_dict(global_state_dict, strict=True)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = (9 - labels).to(self.device)  # 标签翻转 y = 9 - y
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# --------------- 反向梯度 / 更新客户端 ---------------


class SignFlippingClient(BaseClient):
    """
    反向梯度攻击客户端：
    - 先按 BenignClient 方式本地训练得到 local_model
    - 计算本地更新 Δ = local - global
    - 上传 global - scale * Δ，实现“更新方向取反/放大”
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 0.1,
        local_epochs: int = 1,
        scale: float = 1.0,
    ):
        super().__init__(client_id, device)
        self.dataloader = dataloader
        self.lr = lr
        self.local_epochs = local_epochs
        self.scale = scale

    def local_step(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        # 拷贝一份 global，用于之后计算 Δ
        global_sd_device = {k: v.to(self.device) for k, v in global_state_dict.items()}

        model = get_resnet18_cifar10(num_classes=10).to(self.device)
        model.load_state_dict(global_sd_device, strict=True)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        local_sd = model.state_dict()
        poisoned_sd: Dict[str, torch.Tensor] = {}
        for k, v_local in local_sd.items():
            v_global = global_sd_device[k]
            if v_local.is_floating_point():
                delta = v_local - v_global
                poisoned_sd[k] = (v_global - self.scale * delta).cpu().clone()
            else:
                poisoned_sd[k] = v_global.cpu().clone()
        return poisoned_sd
