from __future__ import annotations

"""
从 bn_cifar10.pth 随机划分并生成中心化训练用的 CIFAR-10 BN 特征子集。

假定 bn_cifar10.pth 的结构为：
    {
        "features": Tensor [N, D_bn],  # BN 特征
        "labels":   Tensor [N],        # 对应标签 0..9 （可选，但推荐保留）
    }

用法示例（在 article2 目录下）：
    python -m test_svdd.prepare_cifar10_subset --src bn_cifar10.pth --dst cifar10.pth
"""

import argparse
from pathlib import Path

import torch

from new.utils import extract_bn_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="bn_cifar10.pth", help="原始 BN 特征文件路径")
    parser.add_argument("--dst", type=str, default="cifar10.pth", help="输出子集文件路径")
    parser.add_argument("--num_clients", type=int, default=10, help="逻辑划分的客户端数量")
    parser.add_argument("--per_client", type=int, default=3000, help="从某一份中取出的样本数")
    parser.add_argument("--client_id", type=int, default=0, help="选择哪一份 (0..num_clients-1)")
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="模拟高斯噪声客户端的样本比例（例如 0.2 表示 20%% 样本加噪）",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.5,
        help="高斯噪声标准差",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.is_file():
        raise FileNotFoundError(f"bn_cifar10 源文件不存在: {src_path}")

    data = torch.load(src_path, map_location="cpu")

    # 兼容 collector.py 生成的格式：
    # {
    #   "bn_params_history": [ state_dict_like_bn_only, ... ],
    #   ...
    # }
    if isinstance(data, dict) and "bn_params_history" in data:
        bn_list = data["bn_params_history"]
        # 将每个 BN state_dict 展开为一条特征向量
        feats = [extract_bn_features(sd) for sd in bn_list]
        X = torch.stack(feats, dim=0)
        y = None
    elif isinstance(data, dict):
        if "features" in data:
            X = data["features"]
        elif "x" in data:
            X = data["x"]
        else:
            raise KeyError("bn_cifar10.pth 中未找到 'features' 或 'x' 键，也不是 collector.py 的 bn_params_history 结构。")

        if "labels" in data:
            y = data["labels"]
        elif "y" in data:
            y = data["y"]
        else:
            y = None
    else:
        # 兼容直接保存 Tensor 的情况
        X = torch.as_tensor(data)
        y = None

    N, d_bn = X.shape
    print(f"[INFO] 加载到 BN 特征: N={N}, D_bn={d_bn}")

    # 随机打乱并均匀划分为 num_clients 份
    perm = torch.randperm(N)
    X = X[perm]
    if y is not None:
        y = y[perm]

    split_size = N // args.num_clients
    if args.client_id < 0 or args.client_id >= args.num_clients:
        raise ValueError(f"client_id 必须在 [0, {args.num_clients - 1}] 范围内")

    start = args.client_id * split_size
    end = N if args.client_id == args.num_clients - 1 else (args.client_id + 1) * split_size
    X_client = X[start:end]
    y_client = None if y is None else y[start:end]

    # 只取前 per_client 条
    if X_client.size(0) < args.per_client:
        raise ValueError(f"该份数据不足 {args.per_client} 条，实际 {X_client.size(0)} 条。")
    X_client = X_client[: args.per_client]
    if y_client is not None:
        y_client = y_client[: args.per_client]

    # 构造 20% 高斯噪声样本
    num_noise = int(args.noise_ratio * args.per_client)
    if num_noise > 0:
        X_noisy = X_client.clone()
        idx = torch.randperm(args.per_client)[:num_noise]
        X_noisy[idx] = X_noisy[idx] + torch.randn_like(X_noisy[idx]) * args.noise_sigma

        # 拼成一个中心化数据集：前 0.8 为“正常”，后 0.2 为“高斯噪声客户端”模拟
        X_all = torch.cat([X_client, X_noisy[idx]], dim=0)
        if y_client is not None:
            # 噪声样本标签与原样本相同，攻击只体现在特征空间
            y_all = torch.cat([y_client, y_client[idx]], dim=0)
        else:
            y_all = None

        is_malicious = torch.zeros(X_all.size(0), dtype=torch.long)
        is_malicious[args.per_client :] = 1  # 后半部分视作“恶意特征”
    else:
        X_all = X_client
        y_all = y_client
        is_malicious = torch.zeros(X_all.size(0), dtype=torch.long)

    out = {"features": X_all, "labels": y_all, "is_malicious": is_malicious}
    torch.save(out, args.dst)
    print(f"[INFO] 保存子集到 {args.dst}，总样本数={X_all.size(0)}，其中恶意标记={is_malicious.sum().item()}")


if __name__ == "__main__":
    main()

