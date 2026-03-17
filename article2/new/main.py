from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clients import ATTACK_REGISTRY, BaseClient, BenignClient
from config import FedConfig
from dataset import build_cifar10_dataloaders
from models import build_resnet18
from server import FederatedServer
from utils import extract_bn_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: FedConfig) -> torch.device:
    if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, int, int]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / max(1, total)
    return acc, correct, total


def build_clients(
    config: FedConfig, device: torch.device, loaders: List[DataLoader]
) -> Tuple[List[BaseClient], torch.Tensor]:
    """Create benign and attack clients; return list and ground-truth labels (1=benign, 0=malicious)."""

    num_clients = config.num_clients
    num_benign = config.num_benign
    benign_ids = list(range(num_benign))
    malicious_ids = list(range(num_benign, num_clients))

    gt = torch.zeros(num_clients, dtype=torch.long)
    gt[benign_ids] = 1

    clients: List[BaseClient] = []

    def model_fn():
        return build_resnet18(num_classes=10)

    # Benign
    for cid in benign_ids:
        clients.append(BenignClient(cid, device, config, loaders[cid], model_fn))

    # Malicious
    attack_cls = ATTACK_REGISTRY.get(config.attack_type, None)
    if attack_cls is None:
        raise ValueError(f"Unknown attack_type: {config.attack_type}")
    for cid in malicious_ids:
        if issubclass(attack_cls, BenignClient):
            clients.append(attack_cls(cid, device, config, loaders[cid], model_fn))
        else:
            clients.append(attack_cls(cid, device, config, loaders[cid]))

    return clients, gt


def run_federated(config: FedConfig, use_svdd: bool = True) -> None:
    set_seed(config.seed)
    device = resolve_device(config)

    client_loaders, test_loader = build_cifar10_dataloaders(config)
    clients, gt = build_clients(config, device, client_loaders)

    # Infer BN feature dimension from a temporary model
    tmp_model = build_resnet18(num_classes=10)
    d_bn = extract_bn_features(tmp_model.state_dict()).numel()

    server = FederatedServer(config, d_bn=d_bn, device=device)

    total_rounds = config.total_rounds
    phase1_rounds = config.phase1_rounds

    for r in range(1, total_rounds + 1):
        global_sd = server._state_dict_for_clients()

        client_sds: List[Dict[str, Tensor]] = []
        for c in clients:
            local_sd = c.local_step(global_sd)
            client_sds.append(local_sd)

        if not use_svdd:
            # Pure FedAvg
            K = len(client_sds)
            alpha = torch.full((K,), 1.0 / K)
            from utils import weighted_fedavg

            global_sd = weighted_fedavg(client_sds, alpha)
            server.global_model.load_state_dict(global_sd)
            center_norm = float(0.0)
            z_var = 0.0
            ae_loss = 0.0
            svdd_loss = 0.0
            d = torch.zeros(K)
            M = torch.ones(K)
            alpha_out = alpha
        else:
            if r <= phase1_rounds:
                center_norm, z_var, ae_loss, d, keep_mask = server.phase1_step(r, client_sds)
                svdd_loss = 0.0
                # Phase1 中 M 表示该客户端是否被 trimmed loss 保留（1=参与 AE 损失，0=被剪掉）
                M = keep_mask.float()
                alpha_out = torch.full((len(client_sds),), 1.0 / len(client_sds))
            elif r == phase1_rounds + 1 and server.c is None:
                center_norm, z_var = server.init_center(client_sds)
                svdd_loss = 0.0
                ae_loss = 0.0
                d = torch.zeros(len(client_sds))
                M = torch.ones(len(client_sds))
                alpha_out = torch.full((len(client_sds),), 1.0 / len(client_sds))
            else:
                svdd_round = r - phase1_rounds
                (
                    center_norm,
                    z_var,
                    svdd_loss,
                    _recon_loss,
                    d,
                    M,
                    alpha_out,
                ) = server.phase2_step(svdd_round, client_sds)  
                ae_loss = 0.0

        # Evaluation
        test_acc, correct, total = evaluate(server.global_model, test_loader, device)

        # Monitoring metrics
        z_var_scalar = z_var
        if z_var_scalar < 1e-6 and use_svdd and r > phase1_rounds:
            print(f"[WARN] Encoder collapse detected at round {r}: var={z_var_scalar:.3e}")

        # Ground-truth based TPR / FPR
        benign_mask = gt == 1
        mal_mask = gt == 0
        with torch.no_grad():
            rejected = (M < 0.5)
            if mal_mask.any():
                tpr = float((rejected[mal_mask]).float().mean().item())
            else:
                tpr = 0.0
            if benign_mask.any():
                fpr = float((rejected[benign_mask]).float().mean().item())
            else:
                fpr = 0.0

        # Pretty monitor output
        phase = "AE Warm-up" if r <= phase1_rounds else ("SVDD Filtering" if use_svdd else "FedAvg Baseline")
        print_monitor_round(
            round_idx=r,
            phase=phase,
            num_clients=config.num_clients,
            num_benign=config.num_benign,
            center_norm=center_norm if use_svdd else float("nan"),
            z_var=z_var_scalar,
            ae_loss=ae_loss if r <= phase1_rounds else float("nan"),
            svdd_loss=svdd_loss if (use_svdd and r > phase1_rounds) else float("nan"),
            test_acc=test_acc,
            test_correct=correct,
            test_total=total,
            d=d,
            alpha=alpha_out,
            M=M,
            gt=gt,
            tpr=tpr,
            fpr=fpr,
            show_detection=use_svdd,
        )


def print_monitor_round(
    round_idx: int,
    phase: str,
    num_clients: int,
    num_benign: int,
    center_norm: float,
    z_var: float,
    ae_loss: float,
    svdd_loss: float,
    test_acc: float,
    test_correct: int,
    test_total: int,
    d: Tensor,
    alpha: Tensor,
    M: Tensor,
    gt: Tensor,
    tpr: float,
    fpr: float,
    show_detection: bool,
) -> None:
    num_mal = num_clients - num_benign
    header = f"--- Round {round_idx} ---"
    print(header)
    print("+-----------------------------------------------------------------------------+")
    print(
        f"| AE-SVDD Monitor Round {round_idx:<3d}  Phase {phase:<15}|"  # padded later by spaces
    )
    print(f"|Clients {num_clients:<2d}  Benign {num_benign:<2d}  Malicious {num_mal:<2d}".ljust(77) + "|")
    print("+-----------------------------------------------------------------------------+")

    center_str = f"{center_norm:.6f}" if not np.isnan(center_norm) else "N/A"
    print(f"| Center L2-Norm           | {center_str:<47}|")
    print(f"| Z-Space Variance         | {z_var:<47.6f}|")
    if round_idx <= 50:
        print(f"| AE L1-Loss               | {ae_loss:<47.6f}|")
    else:
        print(f"| SVDD Loss                | {svdd_loss:<47.6f}|")
    acc_str = f"{test_acc:.4f}  ({test_correct}/{test_total})"
    print(f"| Test Accuracy            | {acc_str:<47}|")
    print("+-----------------------------------------------------------------------------+")

    if show_detection:
        # Detection / trimming summary
        print()
        if phase.startswith("AE Warm-up"):
            title = "AE Trimmed Loss"
        else:
            title = "Detection (Phase 2)"
        print("+------------------------+--------------+--------------+")
        print(f"| {title:<22} |       Benign |    Malicious |")
        print("+------------------------+--------------+--------------+")

        benign_mask = gt == 1
        mal_mask = gt == 0
        rejected = (M < 0.5)

        def avg_or_zero(vals: Tensor) -> float:
            if vals.numel() == 0:
                return 0.0
            return float(vals.mean().item())

        dist_ben = avg_or_zero(d[benign_mask])
        dist_mal = avg_or_zero(d[mal_mask])
        w_ben = avg_or_zero(alpha[benign_mask])
        w_mal = avg_or_zero(alpha[mal_mask])

        tpr_str = "-" if mal_mask.sum() == 0 else f"{tpr:.4f}"
        fpr_str = "-" if benign_mask.sum() == 0 else f"{fpr:.4f}"

        if phase.startswith("AE Warm-up"):
            print(f"| Loss (avg)             | {dist_ben:12.6f} | {dist_mal:12.6f} |")
        else:
            print(f"| Dist (avg)             | {dist_ben:12.6f} | {dist_mal:12.6f} |")
        print(f"| Weight (avg)           | {w_ben:12.6f} | {w_mal:12.6f} |")
        print(f"| TPR (mal. reject)      | {'-':>12} | {tpr_str:12} |")
        print(f"| FPR (ben. reject)      | {fpr_str:12} | {'-':>12} |")
        print("+------------------------+--------------+--------------+")

        # Per-client table
        print("+-------+----------+----------------+----------------+-------+")
        if phase.startswith("AE Warm-up"):
            col_name = "AE L1-Loss"
        else:
            col_name = "Dist"
        print(f"|    ID | Type     | {col_name:>14} |          Alpha |     M |")
        print("+-------+----------+----------------+----------------+-------+")
        for i in range(num_clients):
            ctype = "Benign" if gt[i].item() == 1 else "Mal"
            print(
                f"| {i:5d} | {ctype:<8} | {d[i].item():14.6f} | {alpha[i].item():14.6f} | {int(M[i].item() >= 0.5):5d} |"
            )
        print("+-------+----------+----------------+----------------+-------+")


if __name__ == "__main__":
    cfg = FedConfig()
    run_federated(cfg, use_svdd=True)

