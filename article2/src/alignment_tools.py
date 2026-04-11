from __future__ import annotations

"""
Alignment utilities for cross-framework experiments.

This module reuses the *exact* data partitioning and model construction
logic from the local AE-SVDD framework to:

1. 导出 Dirichlet 划分映射（alpha=5.0, seed=42, num_clients=50）。
2. 导出 CIFAR-10 ResNet18 (no maxpool) 的初始权重，用于对齐外部代码库。

用法示例（在 article2 项目根目录下）:

  # 1) 生成 Dirichlet 映射 JSON
  python -m src.alignment_tools \
      --export-dirichlet-mapping \
      --output experiment/dirichlet_cifar10_alpha5_seed42_K50.json

  # 2) 导出 ResNet18 CIFAR-10 初始权重
  python -m src.alignment_tools \
      --export-resnet18-init \
      --output experiment/resnet18_cifar10_init.pth
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch

try:
    from .config import FedConfig
    from .models import resnet18_cifar10
    from .tasks import Cifar10Task
    from .tasks import _client_index_lists_dirichlet_strict, _dataset_train_labels
except ImportError:  # 允许在 src/ 内直接 python alignment_tools.py
    from config import FedConfig
    from models import resnet18_cifar10
    from tasks import Cifar10Task
    from tasks import _client_index_lists_dirichlet_strict, _dataset_train_labels

from torchvision import datasets, transforms


def _build_cifar10_train_dataset(cfg: FedConfig):
    """
    使用与 Cifar10Task 完全一致的变换和根目录逻辑，
    但只构造 train_dataset（避免触发不必要的 test 下载）。
    """
    task = Cifar10Task()
    root = task.data_subdir(cfg)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )
    return train_dataset


def export_dirichlet_mapping(
    output_path: Path,
    num_clients: int = 50,
    alpha: float = 5.0,
    seed: int = 42,
) -> None:
    """
    调用本地 tasks._client_index_lists_dirichlet_strict 生成
    CIFAR-10 的 Dirichlet(alpha=5.0, seed=42) 客户端样本索引映射。

    输出 JSON 结构:
      {
        "meta": {
          "dataset": "cifar10",
          "num_clients": 50,
          "alpha": 5.0,
          "seed": 42
        },
        "client_indices": {
          "0": [idx0, idx1, ...],
          "1": [...],
          ...
        }
      }
    """
    cfg = FedConfig()
    cfg.num_clients = num_clients
    cfg.dirichlet_alpha = alpha
    cfg.seed = seed

    train_dataset = _build_cifar10_train_dataset(cfg)
    labels = _dataset_train_labels(train_dataset)

    client_index_lists: List[List[int]] = _client_index_lists_dirichlet_strict(
        labels=labels,
        num_clients=num_clients,
        num_classes=10,
        alpha=float(alpha),
        seed=seed,
    )

    payload = {
        "meta": {
            "dataset": "cifar10",
            "num_clients": int(num_clients),
            "alpha": float(alpha),
            "seed": int(seed),
            "num_samples": int(labels.numel()),
        },
        "client_indices": {
            str(i): [int(idx) for idx in idx_list]
            for i, idx_list in enumerate(client_index_lists)
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[alignment_tools] Saved Dirichlet mapping -> {output_path.resolve()}")


def export_resnet18_cifar10_init(output_path: Path) -> None:
    """
    按照本地任务定义构造 resnet18_cifar10（无预训练、无 maxpool），
    并导出其初始 state_dict。

    外部框架可以直接:
      - torch.load(path)
      - model.load_state_dict(ckpt)
    来对齐 Round 0 的全局权重。
    """
    # 与 Cifar10Task.build_model 完全一致
    model = resnet18_cifar10(num_classes=10)
    state = model.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"[alignment_tools] Saved resnet18_cifar10 init weights -> {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alignment tools for cross-framework FL experiments.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--export-dirichlet-mapping",
        action="store_true",
        help="导出 CIFAR-10 Dirichlet(alpha=5.0, seed=42, K=50) 客户端索引 JSON。",
    )
    group.add_argument(
        "--export-resnet18-init",
        action="store_true",
        help="导出 resnet18_cifar10 初始权重 state_dict (Round 0)。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（.json / .pth）。",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=50,
        help="客户端数量（默认 50，仅用于 Dirichlet 导出）。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Dirichlet alpha（默认 5.0，仅用于 Dirichlet 导出）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42，仅用于 Dirichlet 导出）。",
    )

    args = parser.parse_args()
    out_path = Path(args.output)

    if args.export_dirichlet_mapping:
        export_dirichlet_mapping(
            output_path=out_path,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
        )
    elif args.export_resnet18_init:
        export_resnet18_cifar10_init(out_path)
    else:
        raise RuntimeError("Unknown mode.")


if __name__ == "__main__":
    main()

