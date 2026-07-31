from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..batched_clients import BatchedClientExecutor, train_clients_batched_or_serial
from ..clients import ATTACK_REGISTRY, BaseClient, BenignClient, mixed_attack_for_client
from ..config import FedConfig, normalize_attack_name, normalize_defense_name
from ..defenses import DEFENSE_REGISTRY
from ..defenses.svdd import SVDDDefense
from ..reporting.console import print_round_event
from ..tasks import get_task
from .contracts import PipelineContext
from .stages import (
    AttackStage,
    ClientStage,
    ClientTrainStage,
    ConfigStage,
    DataStage,
    DefenseStage,
    EvaluationStage,
    OutputStage,
    RoundPipeline,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: FedConfig) -> torch.device:
    if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
    channels_last: bool = False,
) -> Tuple[float, int, int]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if channels_last and x.ndim == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
                logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / max(1, total)
    return acc, correct, total


def _flatten_float_state_delta(
    before_sd: Dict[str, Tensor],
    after_sd: Dict[str, Tensor],
) -> Tensor:
    parts: List[Tensor] = []
    for k, before in before_sd.items():
        if not before.is_floating_point():
            continue
        after = after_sd[k]
        parts.append((after.detach().cpu().float() - before.detach().cpu().float()).reshape(-1))
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _flatten_bn_buffer_delta(
    before_sd: Dict[str, Tensor],
    after_sd: Dict[str, Tensor],
) -> Tensor:
    """Flatten BN running-stat deltas for monitoring.

    Tracks how much BatchNorm buffers actually change each round; this is
    important for defenses that aggregate parameter deltas explicitly.
    """
    parts: List[Tensor] = []
    for k, before in before_sd.items():
        if not before.is_floating_point():
            continue
        if not (
            ("bn" in k.lower() and k.endswith("running_mean"))
            or ("bn" in k.lower() and k.endswith("running_var"))
        ):
            continue
        after = after_sd[k]
        parts.append((after.detach().cpu().float() - before.detach().cpu().float()).reshape(-1))
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _client_delta_norms(
    global_sd: Dict[str, Tensor],
    client_sds: List[Dict[str, Tensor]],
) -> Tensor:
    norms: List[Tensor] = []
    for sd in client_sds:
        d = _flatten_float_state_delta(global_sd, sd)
        norms.append(d.norm(p=2))
    if not norms:
        return torch.zeros(0, dtype=torch.float32)
    return torch.stack(norms).float()

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

    # Matrix clients execute serially, so they may share one training model.
    # Direct runs can retain historical model-construction RNG consumption.
    shared_client_model = None
    if config.reuse_client_model:
        shared_client_model = task.build_model().to(device)
        if config.channels_last and device.type == "cuda":
            shared_client_model = shared_client_model.to(memory_format=torch.channels_last)

    def model_fn():
        if shared_client_model is not None:
            return shared_client_model
        model = task.build_model().to(device)
        if config.channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        return model

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
    if defense_name == "mk":
        return int(
            config.krum_num_byzantine
            if config.krum_num_byzantine is not None
            else max(0, config.num_clients - config.num_benign)
        )
    if defense_name == "tm":
        if config.trimmed_mean_num_byzantine is not None:
            return int(config.trimmed_mean_num_byzantine)
        return max(0, config.num_clients - config.num_benign)
    return 0


def _apply_lie_attack(
    config: FedConfig,
    defense_name: str,
    global_sd: Dict[str, Tensor],
    client_sds: List[Dict[str, Tensor]],
) -> None:
    """Rewrite all malicious uploads using ALIE/LIE: delta = mu + z * sigma."""
    if config.attack_type not in {"lie", "mix"}:
        return

    n = int(config.num_clients)
    m = max(0, n - int(config.num_benign))
    if m <= 0:
        return

    benign_n = max(1, n - m)
    lie_ids = (
        list(range(config.num_benign, n))
        if config.attack_type == "lie"
        else [
            cid for cid in range(config.num_benign, n)
            if mixed_attack_for_client(config, cid) == "lie"
        ]
    )
    if not lie_ids:
        return
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

    for cid in lie_ids:
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


def _feature_dimension(context: PipelineContext) -> int:
    config = context.config
    if context.defense_name != "svdd":
        return 1
    feature_mode = str(config.svdd_feature_mode).lower().strip()
    if feature_mode == "fixed_projection":
        dimension = int(config.param_descriptor_dim)
        if dimension not in (4096, 8192):
            raise ValueError("param_descriptor_dim must be 4096 or 8192")
        return dimension
    if feature_mode != "task":
        raise ValueError(
            f"Unknown svdd_feature_mode {config.svdd_feature_mode!r}; "
            "use 'task' or 'fixed_projection'"
        )
    with torch.random.fork_rng(devices=[]):
        state = context.task.build_model().state_dict()
    return int(context.task.extract_svdd_features(config, state).numel())


def _create_defense(context: PipelineContext):
    defense_cls = DEFENSE_REGISTRY.get(context.defense_name)
    if defense_cls is None:
        raise ValueError(f"Unknown defense_type: {context.defense_name}")

    def model_fn():
        return context.task.build_model()

    kwargs: Dict[str, Any] = {
        "config": context.config,
        "d_bn": _feature_dimension(context),
        "device": context.device,
        "model_fn": model_fn,
    }
    if context.defense_name == "svdd":
        kwargs["svdd_feature_extractor"] = lambda state: (
            context.task.extract_svdd_features(context.config, state)
        )
    return defense_cls(**kwargs), model_fn


def _round_diagnostics(context: PipelineContext) -> Dict[str, float]:
    if not context.config.round_diagnostics:
        return {}
    new_global = context.defense_result.global_state
    global_delta = _flatten_float_state_delta(context.global_state, new_global)
    client_norms = _client_delta_norms(context.global_state, context.client_states)
    benign = context.ground_truth == 1
    malicious = context.ground_truth == 0
    values: Dict[str, float] = {
        "upd_l2": float(global_delta.norm(p=2).item()),
        "upd_linf": float(global_delta.abs().max().item()),
        "upd_mean_abs": float(global_delta.abs().mean().item()),
        "upd_nonzero_ratio": float((global_delta != 0).float().mean().item()),
        "ben_norm_mean": float(client_norms[benign].mean().item()) if benign.any() else 0.0,
        "mal_norm_mean": float(client_norms[malicious].mean().item()) if malicious.any() else 0.0,
        "cos_ben": 0.0,
        "cos_mal": 0.0,
    }
    bn_delta = _flatten_bn_buffer_delta(context.global_state, new_global)
    values.update(
        {
            "bn_upd_l2": float(bn_delta.norm(p=2).item()),
            "bn_upd_linf": float(bn_delta.abs().max().item()),
            "bn_upd_mean_abs": float(bn_delta.abs().mean().item()),
            "bn_upd_nonzero_ratio": float((bn_delta != 0).float().mean().item()),
        }
    )
    for label, mask in (("ben", benign), ("mal", malicious)):
        if not bool(mask.any().item()):
            continue
        indices = torch.where(mask)[0].tolist()
        mean_delta = torch.stack(
            [
                _flatten_float_state_delta(context.global_state, context.client_states[i])
                for i in indices
            ],
            dim=0,
        ).mean(dim=0)
        values[f"cos_{label}"] = float(
            torch.nn.functional.cosine_similarity(
                global_delta.unsqueeze(0), mean_delta.unsqueeze(0), dim=1
            ).item()
        )
    return values


def _build_round_event(context: PipelineContext) -> Dict[str, Any]:
    result = context.defense_result
    accepted = result.accepted_mask.detach().cpu().reshape(-1) >= 0.5
    ground_truth = context.ground_truth.detach().cpu().reshape(-1)
    benign = ground_truth == 1
    malicious = ground_truth == 0
    rejected = ~accepted
    tp = int((rejected & malicious).sum().item())
    fp = int((rejected & benign).sum().item())
    tn = int((accepted & benign).sum().item())
    fn = int((accepted & malicious).sum().item())
    tpr = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))
    total_clients = max(1, tp + fp + tn + fn)
    diagnostics = _round_diagnostics(context)
    result.diagnostics.update(diagnostics)
    return {
        "round": context.round_idx,
        "phase": result.phase,
        "defense": context.defense_name,
        "stats": result,
        "ground_truth": ground_truth,
        "test_acc": context.evaluation["accuracy"],
        "test_correct": context.evaluation["correct"],
        "test_total": context.evaluation["total"],
        "tpr": tpr,
        "fpr": fpr,
        "dar": float((tp + tn) / total_clients),
        "dpr": float(tp / max(1, tp + fp)),
        "rr": float(tp / max(1, tp + fn)),
        "reject_rate": float(rejected.float().mean().item()),
        "diagnostics": diagnostics,
    }


def run_pipeline(context: PipelineContext) -> PipelineContext:
    """Execute the full Config/Data/Client/Round pipeline for one matrix combo."""

    config = context.config
    set_seed(config.seed)
    config.attack_type = normalize_attack_name(config.attack_type)
    config.defense_type = normalize_defense_name(config.defense_type)
    config.aggregation_method = normalize_defense_name(config.aggregation_method)
    context.attack_name = normalize_attack_name(context.attack_name)
    context.defense_name = normalize_defense_name(context.defense_name)
    context.device = resolve_device(config)
    if context.device.type == "cuda" and (config.use_amp or config.channels_last):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    context.task = get_task(config)
    config.num_classes = context.task.num_classes
    ConfigStage().run(context)
    DataStage(context.task.build_dataloaders).run(context)
    ClientStage(build_clients).run(context)
    context.defense, model_fn = _create_defense(context)
    if int(config.client_batch_group_size) > 1:
        context.batched_executor = BatchedClientExecutor(
            config, context.device, model_fn
        )

    RoundPipeline(
        ClientTrainStage(train_clients_batched_or_serial),
        AttackStage(_apply_lie_attack),
        DefenseStage(),
        EvaluationStage(evaluate, _build_round_event),
        OutputStage(print_round_event),
    ).run(context)
    return context


__all__ = [
    "build_clients",
    "evaluate",
    "resolve_device",
    "run_pipeline",
    "set_seed",
]
