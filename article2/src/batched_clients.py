from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

try:
    from .clients import BenignClient
    from .config import FedConfig
except ImportError:
    from clients import BenignClient
    from config import FedConfig


class BatchedClientExecutor:
    """Train independent benign clients in one vmap call per local step."""

    def __init__(
        self,
        config: FedConfig,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        self.config = config
        self.device = device
        self.group_size = int(config.client_batch_group_size)
        if self.group_size < 2:
            raise ValueError("client_batch_group_size must be at least 2.")
        if self.device.type != "cuda":
            raise ValueError("Batched client execution currently requires CUDA.")
        if config.use_amp:
            raise ValueError("Batched client execution does not yet support AMP.")

        # Constructing the functional model must not perturb experiment RNG.
        with torch.random.fork_rng(devices=[]):
            functional_model = model_fn()
        self.functional_model = functional_model.to("meta").train()
        self.parameter_names = tuple(name for name, _ in functional_model.named_parameters())
        self.buffer_names = tuple(name for name, _ in functional_model.named_buffers())
        self._parameter_set = set(self.parameter_names)
        self._buffer_set = set(self.buffer_names)

        def loss_fn(
            params: Dict[str, Tensor],
            buffers: Dict[str, Tensor],
            x: Tensor,
            y: Tensor,
        ) -> Tensor:
            logits = functional_call(self.functional_model, (params, buffers), (x,))
            return F.cross_entropy(logits, y)

        self._batched_grad = vmap(
            grad(loss_fn),
            in_dims=(0, 0, 0, 0),
            randomness="different",
        )

    def can_batch(self, clients: Sequence[BenignClient]) -> bool:
        if len(clients) != self.group_size:
            return False
        # Only the exact benign implementation is enabled. Attack subclasses
        # retain their existing local_step and postprocessing semantics.
        if any(type(client) is not BenignClient for client in clients):
            return False
        loader_lengths = {len(client.loader) for client in clients}
        dataset_lengths = {len(client.loader.dataset) for client in clients}
        return (
            int(self.config.local_epochs) > 0
            and loader_lengths != {0}
            and len(loader_lengths) == 1
            and len(dataset_lengths) == 1
        )

    def _stack_state(
        self,
        global_state_dict: Dict[str, Tensor],
        names: Sequence[str],
        count: int,
        *,
        requires_grad: bool,
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for name in names:
            value = global_state_dict[name].detach()
            stacked = value.unsqueeze(0).expand(count, *value.shape).clone()
            if requires_grad:
                stacked.requires_grad_()
            out[name] = stacked
        return out

    def train_group(
        self,
        clients: Sequence[BenignClient],
        global_state_dict: Dict[str, Tensor],
        reference_state_dict: Dict[str, Tensor],
    ) -> List[Dict[str, Tensor]]:
        if not self.can_batch(clients):
            raise ValueError("Client group is not compatible with batched execution.")

        count = len(clients)
        params = self._stack_state(
            global_state_dict,
            self.parameter_names,
            count,
            requires_grad=True,
        )
        buffers = self._stack_state(
            global_state_dict,
            self.buffer_names,
            count,
            requires_grad=False,
        )
        momentum = {name: torch.zeros_like(value) for name, value in params.items()}

        lr = float(self.config.client_lr)
        momentum_factor = float(self.config.client_momentum)
        weight_decay = float(self.config.client_weight_decay)

        for _ in range(int(self.config.local_epochs)):
            iterators = [iter(client.loader) for client in clients]
            for _step in range(len(clients[0].loader)):
                xs: List[Tensor] = []
                ys: List[Tensor] = []
                for client, iterator in zip(clients, iterators):
                    x, y = next(iterator)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    x, y = client._transform_batch(x, y)
                    xs.append(x)
                    ys.append(y)

                x_group = torch.stack(xs, dim=0)
                y_group = torch.stack(ys, dim=0)
                grads = self._batched_grad(params, buffers, x_group, y_group)
                with torch.no_grad():
                    for name, value in params.items():
                        update = grads[name]
                        if weight_decay != 0.0:
                            update = update.add(value, alpha=weight_decay)
                        momentum[name].mul_(momentum_factor).add_(update)
                        value.add_(momentum[name], alpha=-lr)

        uploads: List[Dict[str, Tensor]] = []
        for client_idx, client in enumerate(clients):
            local_state: Dict[str, Tensor] = {}
            for name in reference_state_dict:
                if name in self._parameter_set:
                    local_state[name] = params[name][client_idx]
                elif name in self._buffer_set:
                    local_state[name] = buffers[name][client_idx]
                else:
                    raise KeyError(f"State entry {name!r} is neither a parameter nor a buffer.")
            uploads.append(client._postprocess_upload(reference_state_dict, local_state))

        del params, buffers, momentum
        return uploads


def train_clients_batched_or_serial(
    clients: Sequence[BenignClient],
    global_state_dict: Dict[str, Tensor],
    reference_state_dict: Dict[str, Tensor],
    executor: BatchedClientExecutor,
) -> List[Dict[str, Tensor]]:
    """Preserve client order while batching compatible full-size groups."""

    outputs: List[Dict[str, Tensor]] = []
    idx = 0
    while idx < len(clients):
        group = clients[idx : idx + executor.group_size]
        if executor.can_batch(group):
            outputs.extend(executor.train_group(group, global_state_dict, reference_state_dict))
            idx += executor.group_size
            continue
        client = clients[idx]
        outputs.append(
            client.local_step(
                global_state_dict,
                reference_state_dict=reference_state_dict,
            )
        )
        idx += 1
    return outputs
