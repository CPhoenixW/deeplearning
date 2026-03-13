# -*- coding: utf-8 -*-
"""
联邦鲁棒聚合框架 - 主入口
8 个 BenignClient + 2 个 LabelFlippingClient，若干轮通信，两阶段 AE-SVDD 聚合。
"""

import os
from datetime import datetime

import torch
from typing import List, Optional

from dataset import get_client_dataloaders
from clients import BenignClient, LabelFlippingClient, BaseClient, GaussianNoiseClient, SignFlippingClient
from server import FederatedServer
from models import LATENT_DIM


def main(
    use_svdd: bool = True,
    attack_type: str = "gaussian_noise",
    tag: Optional[str] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 攻击手段通过超参数配置：["label_flipping", "gaussian_noise", "sign_flipping"]
    num_clients = 10
    num_benign = 7
    num_gaussian_poison = 3  # 这里表示恶意客户端数量
    total_rounds = 300
    # ground_truth: 1=良性, 0=恶意（仅监控，不参与训练与权重）
    ground_truth_labels: List[int] = [1] * num_benign + [0] * num_gaussian_poison

    client_loaders, test_loader, _ = get_client_dataloaders(
        num_clients=num_clients,
        batch_size=32,
        root="./data",
        seed=42,
        num_workers=0,
    )

    clients: List[BaseClient] = []
    for i in range(num_benign):
        clients.append(
            BenignClient(
                client_id=i,
                dataloader=client_loaders[i],
                device=device,
                lr=0.1,
                local_epochs=1,
            )
        )
    # 根据 attack_type 构造恶意客户端
    for i in range(num_benign, num_clients):
        if attack_type == "label_flipping":
            clients.append(
                LabelFlippingClient(
                    client_id=i,
                    dataloader=client_loaders[i],
                    device=device,
                    lr=0.1,
                    local_epochs=1,
                )
            )
        elif attack_type == "gaussian_noise":
            clients.append(
                GaussianNoiseClient(
                    client_id=i,
                    device=device,
                    sigma=0.6,
                )
            )
        elif attack_type == "sign_flipping":
            clients.append(
                SignFlippingClient(
                    client_id=i,
                    dataloader=client_loaders[i],
                    device=device,
                    lr=0.1,
                    local_epochs=1,
                    scale=1.0,
                )
            )
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

    server = FederatedServer(device=device, lr_ae=1e-3, test_loader=test_loader)
    server.use_center_momentum = False
    # 告知 Server 哪些客户端是良性，确保“只有良性客户端参数才能更新 SVDD 模型”
    server.set_ground_truth_labels(ground_truth_labels)
    global_sd = server.global_model.state_dict()

    # 按日期时间命名本次训练的 log 文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag is not None else ""
    log_path = os.path.join(log_dir, f"train_log_{timestamp}{suffix}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        # 记录本次实验的关键配置，便于离线分析与复现实验
        f.write(f"# use_svdd={use_svdd}\n")
        f.write(
            f"# num_clients={num_clients}, num_benign={num_benign}, num_malicious={num_gaussian_poison}\n"
        )
        f.write(f"# attack_type={attack_type}\n")
        f.write(f"# latent_dim={LATENT_DIM}\n")
        f.write(f"# total_rounds={total_rounds}\n")
        f.write("# round, test_acc, attack_success_rate, false_positive_rate\n")
        for r in range(1, total_rounds + 1):
            # 下发全局模型
            client_states = [client.local_step(copy_global_sd(global_sd)) for client in clients]
            if use_svdd:
                global_sd = server.aggregate(client_states, round_num=r)
            else:
                # 纯 FedAvg 聚合作为 baseline，对照 AE-SVDD 的鲁棒性与准确率
                global_sd = server.aggregate_fedavg(client_states, round_num=r)
            print(f"\n--- Round {r} ---")
            metrics = server.evaluate_system_state(ground_truth_labels, client_states=client_states)

            acc = metrics.get("test_acc", None)
            tpr = metrics.get("tpr", None)
            fpr = metrics.get("fpr", None)
            # 攻击成功率这里定义为：1 - TPR（恶意客户端未被成功截断的比例）
            attack_success = 1.0 - tpr if tpr is not None else None

            def _fmt(x: float) -> str:
                if x is None:
                    return "nan"
                return f"{x:.4f}"

            f.write(
                f"{r}, {_fmt(acc)}, {_fmt(attack_success)}, {_fmt(fpr)}\n"
            )


def copy_global_sd(sd: dict) -> dict:
    """深拷贝 state_dict，避免客户端修改影响服务端。"""
    return {k: v.cpu().clone() for k, v in sd.items()}


if __name__ == "__main__":
    main()
