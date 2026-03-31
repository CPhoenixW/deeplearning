from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from .clients import ATTACK_REGISTRY, BaseClient, BenignClient
    from .config import FedConfig
    from .server import DEFENSE_REGISTRY, BaseServer
    from .tasks import get_task
    from .utils import extract_bn_features
except ImportError:
    from clients import ATTACK_REGISTRY, BaseClient, BenignClient
    from config import FedConfig
    from server import DEFENSE_REGISTRY, BaseServer
    from tasks import get_task
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
    config: FedConfig,
    device: torch.device,
    loaders: List[DataLoader],
    task,
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
        return task.build_model()

    # Benign
    for cid in benign_ids:
        clients.append(BenignClient(cid, device, config, loaders[cid], model_fn))

    # Malicious
    attack_cls = ATTACK_REGISTRY.get(config.attack_type, None)
    if attack_cls is None:
        raise ValueError(f"Unknown attack_type: {config.attack_type}")
    for cid in malicious_ids:
        clients.append(attack_cls(cid, device, config, loaders[cid], model_fn))

    return clients, gt


def resolve_defense_name(config: FedConfig, use_svdd: Optional[bool]) -> str:
    if use_svdd is None:
        return config.defense_type
    return "svdd" if use_svdd else config.aggregation_method


def _default_lie_s(config: FedConfig, defense_name: str) -> int:
    """Infer defense-specific s used in z_max formula when lie_s is not set."""
    if defense_name == "multi_krum":
        return int(
            config.krum_num_byzantine
            if config.krum_num_byzantine is not None
            else max(0, config.num_clients - config.num_benign)
        )
    if defense_name == "trimmed_mean":
        benign_n = max(1, config.num_benign)
        return int(config.trimmed_mean_ratio * benign_n)
    return 0


def _apply_lie_attack(
    config: FedConfig,
    defense_name: str,
    global_sd: Dict[str, Tensor],
    client_sds: List[Dict[str, Tensor]],
) -> None:
    """Rewrite all malicious uploads using ALIE/LIE: delta = mu + z * sigma."""
    if config.attack_type.lower().strip() != "lie_attack":
        return

    n = int(config.num_clients)
    m = max(0, n - int(config.num_benign))
    if m <= 0:
        return

    benign_n = max(1, n - m)
    s = int(config.lie_s) if config.lie_s is not None else _default_lie_s(config, defense_name)
    s = max(0, min(s, benign_n - 1))
    ratio = float(benign_n - s) / float(benign_n)
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    if config.lie_z_override is not None:
        z = float(config.lie_z_override)
    else:
        z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(ratio)).item())

    benign_updates = client_sds[: config.num_benign]
    crafted_delta: Dict[str, Tensor] = {}
    for k, g in global_sd.items():
        g_cpu = g.detach().cpu()
        if not g_cpu.is_floating_point():
            continue
        deltas = torch.stack(
            [(sd[k].detach().cpu().float() - g_cpu.float()) for sd in benign_updates], dim=0
        )
        mu = deltas.mean(dim=0)
        sigma = deltas.std(dim=0, unbiased=False)
        crafted_delta[k] = mu + z * sigma

    for cid in range(config.num_benign, n):
        rewritten: Dict[str, Tensor] = {}
        src = client_sds[cid]
        for k, g in global_sd.items():
            g_cpu = g.detach().cpu()
            if g_cpu.is_floating_point():
                out = g_cpu.float() + crafted_delta[k]
                rewritten[k] = out.to(dtype=g_cpu.dtype).clone()
            else:
                rewritten[k] = src[k].detach().cpu().clone()
        client_sds[cid] = rewritten


def run_federated(
    config: FedConfig,
    use_svdd: Optional[bool] = None,
    collect_metrics: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    set_seed(config.seed)
    device = resolve_device(config)

    task = get_task(config)
    config.num_classes = task.num_classes

    client_loaders, test_loader = task.build_dataloaders(config)
    clients, gt = build_clients(config, device, client_loaders, task)

    tmp_model = task.build_model()
    d_bn = extract_bn_features(tmp_model.state_dict()).numel()

    def model_fn():
        return task.build_model()

    defense_name = resolve_defense_name(config, use_svdd).lower().strip()
    server_cls = DEFENSE_REGISTRY.get(defense_name)
    if server_cls is None:
        raise ValueError(f"Unknown defense_type: {defense_name}")
    server: BaseServer = server_cls(config, d_bn=d_bn, device=device, model_fn=model_fn)

    total_rounds = config.total_rounds

    metrics_history: List[Dict[str, Any]] = []

    for r in range(1, total_rounds + 1):
        global_sd = server.state_dict_for_clients()

        client_sds: List[Dict[str, Tensor]] = []
        for c in clients:
            local_sd = c.local_step(global_sd)
            client_sds.append(local_sd)
        _apply_lie_attack(config, defense_name, global_sd, client_sds)

        stats = server.aggregate(round_idx=r, client_state_dicts=client_sds)
        center_norm = stats.center_norm
        z_var = stats.z_var
        ae_loss = stats.ae_loss
        svdd_loss = stats.svdd_loss
        d = stats.d
        M = stats.m
        alpha_out = stats.alpha
        phase = stats.phase

        # Evaluation
        test_acc, correct, total = evaluate(server.global_model, test_loader, device)

        # Monitoring metrics
        z_var_scalar = z_var
        if z_var_scalar < 1e-6 and defense_name == "svdd":
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
            reject_rate = float(rejected.float().mean().item())

            tp = ((rejected) & mal_mask).sum().item()
            fp = ((rejected) & benign_mask).sum().item()
            tn = ((~rejected) & benign_mask).sum().item()
            fn = ((~rejected) & mal_mask).sum().item()
            total_clients = max(1, tp + fp + tn + fn)
            dar = float((tp + tn) / total_clients)
            dpr = float(tp / max(1, tp + fp))  # Precision
            rr = float(tp / max(1, tp + fn))   # Recall

        # Pretty monitor output
        monitor_items = [("Task", config.task_name)] + list(stats.monitor_items)
        print_monitor_round(
            round_idx=r,
            phase=phase,
            num_clients=config.num_clients,
            num_benign=config.num_benign,
            monitor_items=monitor_items,
            test_acc=test_acc,
            test_correct=correct,
            test_total=total,
            d=d,
            alpha=alpha_out,
            M=M,
            gt=gt,
            tpr=tpr,
            fpr=fpr,
            show_detection=stats.show_detection,
        )

        if collect_metrics:
            metrics_history.append(
                {
                    "round": r,
                    "phase": phase,
                    "test_acc": float(test_acc),
                    "test_correct": int(correct),
                    "test_total": int(total),
                    "malicious_detection_rate": float(tpr),  # TPR
                    "benign_false_positive_rate": float(fpr),  # FPR
                    "dar": float(dar),
                    "dpr": float(dpr),
                    "rr": float(rr),
                    "reject_rate": float(reject_rate),
                }
            )

    if collect_metrics:
        return metrics_history
    return None


def print_monitor_round(
    round_idx: int,
    phase: str,
    num_clients: int,
    num_benign: int,
    monitor_items: List[Tuple[str, str]],
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
    for key, value in monitor_items:
        print(f"| {key:<24} | {value:<47}|")
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
        elif phase.startswith("multi_krum"):
            print(f"| Krum Score (avg)       | {dist_ben:12.6f} | {dist_mal:12.6f} |")
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
        elif phase.startswith("multi_krum"):
            col_name = "Krum Score"
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
    # use_svdd=None 时只看 defense_type，不看 aggregation_method
    cfg.defense_type = "multi_krum"
    run_federated(cfg)