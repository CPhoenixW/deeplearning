# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - 工具函数
BN 特征提取与维度计算，严格满足 I/O 规范。
"""

import torch
from typing import Dict, List, Any


# BN 相关 key 后缀白名单
BN_TENSOR_SUFFIXES = ("weight", "bias", "running_mean", "running_var")


def _is_bn_tensor_key(key: str) -> bool:
    """判断 state_dict 的 key 是否为 BN 层的 weight/bias/running_mean/running_var。"""
    if "bn" not in key.lower():
        return False
    return key.split(".")[-1] in BN_TENSOR_SUFFIXES


def extract_bn_features_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    从单个模型的 state_dict 中提取 BN 特征并拉平为一维向量。

    Process:
        筛选 key 中包含 'bn' 且后缀为 weight, bias, running_mean, running_var 的张量，
        按固定顺序拉平后拼接为一维向量。

    Args:
        state_dict: 模型 state_dict（如 ResNet18）。

    Returns:
        一维张量，形状 (D_bn,)，dtype 与 state_dict 一致。
    """
    parts: List[torch.Tensor] = []
    for k in sorted(state_dict.keys()):
        if not _is_bn_tensor_key(k):
            continue
        t = state_dict[k]
        if t.dim() >= 1:
            parts.append(t.flatten())
        else:
            parts.append(t.reshape(1))
    if not parts:
        return torch.tensor([], dtype=torch.get_default_dtype(), device=next(iter(state_dict.values())).device)
    return torch.cat(parts, dim=0)


def get_bn_feature_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    根据 state_dict 计算 BN 特征维度 D_bn（用于构建 Encoder 输入维度）。

    Args:
        state_dict: 单个模型的 state_dict。

    Returns:
        D_bn: BN 参数展平后的总长度。
    """
    return extract_bn_features_from_state_dict(state_dict).numel()


def build_bn_feature_matrix(client_state_dicts: List[Dict[str, Any]]) -> torch.Tensor:
    """
    从 K 个客户端的 state_dict 构建原始特征矩阵 X。

    Args:
        client_state_dicts: 长度为 K 的列表，每项为 ResNet18 的 state_dict。

    Returns:
        X: 形状 (K, D_bn)，每行为一个客户端的 BN 特征向量。
    """
    rows = [extract_bn_features_from_state_dict(sd) for sd in client_state_dicts]
    return torch.stack(rows, dim=0)
