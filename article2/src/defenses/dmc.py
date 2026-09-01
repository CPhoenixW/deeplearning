"""Paper-faithful FedDMC defense (TDSC 2024).

FedDMC detects malicious federated-learning clients with three server-side
modules described by Mu et al.:

1. DR: PCA projection of client model parameters.
2. BTBCN: binary-tree clustering with noise removal.
3. SEDC: exponential moving-average correction of per-round detections.

The implementation deliberately avoids a clean validation set and does not use
the true number of malicious clients.  PCA is computed from the client Gram
matrix in chunks, so large models do not require materialising one giant
``num_clients x num_parameters`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..config import FedConfig
from ..utils import weighted_fedavg
from .base import BaseDefense, DefenseResult as RoundStats


# The paper/official implementation uses k=10 by default and
# min_cluster_size=3 in the BTBCN experiments.  Keep those as paper defaults
# while allowing an experiment to attach explicit attributes to FedConfig.
_DEFAULT_PCA_DIM = 10
_DEFAULT_MIN_CLUSTER_SIZE = 3
_PCA_CHUNK_SIZE = 65_536


@dataclass
class _ClusterNode:
    """One node in the agglomerative binary clustering tree."""

    node_id: int
    members: Tuple[int, ...]
    centroid: Tensor
    left: Optional["_ClusterNode"] = None
    right: Optional["_ClusterNode"] = None
    merge_cost: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)


class DMCDefense(BaseDefense):
    """FedDMC = PCA dimensionality reduction + BTBCN + SEDC.

    ``dmc_ema_decay`` is reused as the paper's SEDC ensemble coefficient
    :math:`alpha`.  The legacy multi-view DMC knobs remain in ``FedConfig`` for
    compatibility with older experiment files, but they are not part of the
    FedDMC method and therefore are intentionally ignored here.
    """

    defense_name = "dmc"

    def __init__(
        self,
        config: FedConfig,
        d_bn: int,
        device: torch.device,
        model_fn: Callable[[], nn.Module],
    ) -> None:
        super().__init__(config, d_bn, device, model_fn)
        self._trust: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Module 1: DR (PCA)
    # ------------------------------------------------------------------
    @staticmethod
    def _state_is_finite(
        state: Dict[str, Tensor], parameter_names: Sequence[str]
    ) -> bool:
        return all(
            name in state and bool(torch.isfinite(state[name]).all().item())
            for name in parameter_names
        )

    def _pca_scores(
        self,
        client_state_dicts: Sequence[Dict[str, Tensor]],
        client_indices: Sequence[int],
        k: int,
    ) -> Tensor:
        """Return PCA coordinates for selected clients using a chunked Gram matrix.

        Let X be the client-by-parameter matrix after centering every model
        parameter coordinate across clients.  Standard PCA scores can be
        obtained from ``X X^T``.  Accumulating that Gram matrix parameter chunk
        by parameter chunk is algebraically equivalent to flattening every full
        model first, while using substantially less peak memory.
        """

        count = len(client_indices)
        if count == 0:
            return torch.zeros((0, 1), dtype=torch.float32)
        if count == 1:
            return torch.zeros((1, 1), dtype=torch.float32)

        gram = torch.zeros((count, count), dtype=torch.float64)
        for name in self.param_names:
            reference = client_state_dicts[client_indices[0]][name]
            size = int(reference.numel())
            for start in range(0, size, _PCA_CHUNK_SIZE):
                end = min(size, start + _PCA_CHUNK_SIZE)
                block = torch.stack(
                    [
                        client_state_dicts[index][name]
                        .detach()
                        .cpu()
                        .reshape(-1)[start:end]
                        .to(dtype=torch.float64)
                        for index in client_indices
                    ],
                    dim=0,
                )
                block = block - block.mean(dim=0, keepdim=True)
                gram.add_(block @ block.T)

        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]

        max_components = max(1, min(int(k), count - 1))
        positive = int((eigenvalues > 1e-12).sum().item())
        if positive == 0:
            return torch.zeros((count, 1), dtype=torch.float32)
        components = min(max_components, positive)
        values = eigenvalues[:components]
        vectors = eigenvectors[:, :components]
        # X = U S V^T, therefore PCA sample coordinates are U S.
        scores = vectors * torch.sqrt(values).view(1, -1)
        return scores.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Module 2: BTBCN
    # ------------------------------------------------------------------
    @staticmethod
    def _ward_cost(a: _ClusterNode, b: _ClusterNode) -> float:
        """Ward merge criterion used by the authors' public implementation."""

        diff = a.centroid - b.centroid
        scale = float(a.size * b.size) / float(a.size + b.size)
        return scale * float(torch.dot(diff, diff).item())

    @classmethod
    def _build_binary_tree(cls, points: Tensor) -> _ClusterNode:
        """Construct the hierarchical binary tree by repeated closest merging."""

        count = int(points.shape[0])
        if count < 1:
            raise ValueError("BTBCN requires at least one point")
        active: Dict[int, _ClusterNode] = {
            index: _ClusterNode(
                node_id=index,
                members=(index,),
                centroid=points[index].detach().cpu().float().clone(),
            )
            for index in range(count)
        }
        next_id = count

        while len(active) > 1:
            ids = sorted(active)
            best_pair: Optional[Tuple[int, int]] = None
            best_key: Optional[Tuple[float, int, int]] = None
            for offset, left_id in enumerate(ids[:-1]):
                left = active[left_id]
                for right_id in ids[offset + 1 :]:
                    right = active[right_id]
                    cost = cls._ward_cost(left, right)
                    key = (cost, left_id, right_id)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (left_id, right_id)

            assert best_pair is not None and best_key is not None
            left_id, right_id = best_pair
            left = active.pop(left_id)
            right = active.pop(right_id)
            size = left.size + right.size
            centroid = (
                left.centroid * float(left.size)
                + right.centroid * float(right.size)
            ) / float(size)
            merged = _ClusterNode(
                node_id=next_id,
                members=tuple(sorted(left.members + right.members)),
                centroid=centroid,
                left=left,
                right=right,
                merge_cost=float(best_key[0]),
            )
            active[next_id] = merged
            next_id += 1

        return next(iter(active.values()))

    @staticmethod
    def _condense_tree(
        root: _ClusterNode,
        min_cluster_size: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, int]]:
        """Apply the paper's condensed-tree noise removal and binary decision.

        Children smaller than ``min_cluster_size`` are pruned as noisy points.
        Once both children are large enough, the larger child is considered the
        benign cluster, consistent with the paper's assumption that fewer than
        half of the clients are malicious.  Pruned noisy points are rejected.
        """

        total = root.size
        min_size = max(1, int(min_cluster_size))
        outliers: set[int] = set()
        current: Optional[_ClusterNode] = root

        while (
            current is not None
            and current.left is not None
            and current.right is not None
        ):
            left = current.left
            right = current.right
            left_small = left.size < min_size
            right_small = right.size < min_size
            if not left_small and not right_small:
                break
            if left_small and right_small:
                # No valid binary split remains.  Keep the current subtree as
                # benign rather than inventing a malicious cluster unsupported
                # by the FedDMC rule; already-pruned noisy points stay rejected.
                break
            if left_small:
                outliers.update(left.members)
                current = right
            else:
                outliers.update(right.members)
                current = left

            # FedDMC assumes M < floor(N/2).  If noise pruning would exceed
            # that model, stop pruning and preserve the remaining majority.
            if len(outliers) > total // 2:
                break

        benign: set[int] = set()
        malicious: set[int] = set(outliers)
        left_size = 0
        right_size = 0

        if current is not None:
            if current.left is None or current.right is None:
                benign.update(current.members)
            else:
                left = current.left
                right = current.right
                left_size = left.size
                right_size = right.size
                if left.size > right.size:
                    benign.update(left.members)
                    malicious.update(right.members)
                elif right.size > left.size:
                    benign.update(right.members)
                    malicious.update(left.members)
                else:
                    # The paper only defines the larger cluster as benign.
                    # With an exact tie there is no paper-defined discriminator,
                    # so do not arbitrarily reject half of the clients.
                    benign.update(current.members)

        # Any point not explicitly assigned benign is treated as rejected/noise.
        malicious.update(set(range(total)) - benign)
        benign.difference_update(malicious)

        benign_mask = torch.zeros(total, dtype=torch.float32)
        if benign:
            benign_mask[list(sorted(benign))] = 1.0
        outlier_mask = torch.zeros(total, dtype=torch.float32)
        if outliers:
            outlier_mask[list(sorted(outliers))] = 1.0
        info = {
            "raw_benign": int(benign_mask.sum().item()),
            "raw_malicious": int(total - benign_mask.sum().item()),
            "outliers": len(outliers),
            "left_size": left_size,
            "right_size": right_size,
        }
        return benign_mask, outlier_mask, info

    def _btbcn(
        self,
        points: Tensor,
        min_cluster_size: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, int]]:
        count = int(points.shape[0])
        if count == 0:
            return (
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
                {"raw_benign": 0, "raw_malicious": 0, "outliers": 0, "left_size": 0, "right_size": 0},
            )
        if count < 2 or float(points.abs().max().item()) <= 1e-12:
            benign = torch.ones(count, dtype=torch.float32)
            return benign, torch.zeros_like(benign), {
                "raw_benign": count,
                "raw_malicious": 0,
                "outliers": 0,
                "left_size": count,
                "right_size": 0,
            }
        tree = self._build_binary_tree(points)
        return self._condense_tree(tree, min_cluster_size)

    # ------------------------------------------------------------------
    # Module 3: SEDC and aggregation
    # ------------------------------------------------------------------
    def _sedc(self, raw_benign: Tensor) -> Tuple[Tensor, Tensor]:
        clients = int(raw_benign.numel())
        if self._trust is None or self._trust.numel() != clients:
            self._trust = torch.full((clients,), 0.5, dtype=torch.float32)

        alpha = float(getattr(self.config, "dmc_ema_decay", 0.8))
        alpha = min(max(alpha, 0.0), 1.0 - 1e-8)
        self._trust = (
            alpha * self._trust + (1.0 - alpha) * raw_benign.float()
        ).detach()
        accepted = (self._trust >= 0.5).float()
        return self._trust.clone(), accepted

    @staticmethod
    def _paper_parameter(config: FedConfig, name: str, default: int) -> int:
        value = getattr(config, name, default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return int(default)

    def _aggregate(
        self,
        round_idx: int,
        client_state_dicts: List[Dict[str, Tensor]],
    ) -> RoundStats:
        clients = len(client_state_dicts)
        if clients == 0:
            raise ValueError("FedDMC requires at least one client update")

        valid_mask = torch.tensor(
            [
                self._state_is_finite(state, self.param_names)
                for state in client_state_dicts
            ],
            dtype=torch.bool,
        )
        valid_indices = torch.where(valid_mask)[0].tolist()

        pca_dim = self._paper_parameter(
            self.config, "dmc_pca_dim", _DEFAULT_PCA_DIM
        )
        min_cluster_size = self._paper_parameter(
            self.config,
            "dmc_min_cluster_size",
            _DEFAULT_MIN_CLUSTER_SIZE,
        )

        raw_benign = torch.zeros(clients, dtype=torch.float32)
        outlier_full = torch.zeros(clients, dtype=torch.float32)
        pca_norm = torch.zeros(clients, dtype=torch.float32)
        cluster_info = {
            "raw_benign": 0,
            "raw_malicious": clients,
            "outliers": 0,
            "left_size": 0,
            "right_size": 0,
        }

        if valid_indices:
            reduced = self._pca_scores(client_state_dicts, valid_indices, pca_dim)
            local_benign, local_outliers, cluster_info = self._btbcn(
                reduced, min_cluster_size
            )
            raw_benign[valid_indices] = local_benign
            outlier_full[valid_indices] = local_outliers
            pca_norm[valid_indices] = reduced.norm(dim=1)

        # Non-finite uploads are never allowed through SEDC.
        trust, accepted = self._sedc(raw_benign)
        accepted[~valid_mask] = 0.0
        trust[~valid_mask] = 0.0
        if self._trust is not None:
            self._trust[~valid_mask] = 0.0

        accepted_indices = torch.where(accepted >= 0.5)[0].tolist()
        if not accepted_indices:
            # Degenerate fallback: if SEDC rejects everybody, preserve the
            # current global model instead of aggregating an undefined set.
            new_global_state = self.state_dict_for_clients()
            participant_weights = torch.zeros(clients, dtype=torch.float32)
        else:
            participant_weights = torch.zeros(clients, dtype=torch.float32)
            participant_weights[accepted_indices] = 1.0 / float(
                len(accepted_indices)
            )
            selected_states = [client_state_dicts[index] for index in accepted_indices]
            selected_weights = torch.full(
                (len(accepted_indices),),
                1.0 / float(len(accepted_indices)),
                dtype=torch.float32,
            )
            new_global_state = weighted_fedavg(
                selected_states,
                selected_weights,
                device=self.aggregation_device,
            )
            self.global_model.load_state_dict(new_global_state)

        anomaly_score = 1.0 - trust
        kept = int((accepted >= 0.5).sum().item())
        alpha = float(getattr(self.config, "dmc_ema_decay", 0.8))

        return RoundStats(
            center_norm=float("nan"),
            z_var=float(torch.var(anomaly_score).item()) if clients > 1 else 0.0,
            ae_loss=float("nan"),
            svdd_loss=float("nan"),
            d=anomaly_score.detach().cpu(),
            m=accepted.detach().cpu(),
            alpha=participant_weights.detach().cpu(),
            phase="dmc | PCA + BTBCN + SEDC",
            show_detection=True,
            monitor_items=[
                ("Defense", "FedDMC (TDSC 2024)"),
                ("PCA dimension", str(min(pca_dim, max(1, len(valid_indices) - 1)))),
                ("min_cluster_size", str(min_cluster_size)),
                ("SEDC alpha", f"{alpha:.4f}"),
                ("BTBCN benign", f"{cluster_info['raw_benign']}/{len(valid_indices)}"),
                ("BTBCN outliers", str(cluster_info["outliers"])),
                ("Kept clients", f"{kept}/{clients}"),
            ],
            server_metrics={
                "feddmc_pca_dim": int(pca_dim),
                "feddmc_min_cluster_size": int(min_cluster_size),
                "feddmc_sedc_alpha": float(alpha),
                "feddmc_raw_benign": int(cluster_info["raw_benign"]),
                "feddmc_raw_malicious": int(cluster_info["raw_malicious"]),
                "feddmc_outliers": int(cluster_info["outliers"]),
                "feddmc_valid_clients": int(valid_mask.sum().item()),
            },
            participant_metrics={
                "feddmc_trust": trust.detach().cpu(),
                "feddmc_raw_benign": raw_benign.detach().cpu(),
                "feddmc_outlier": outlier_full.detach().cpu(),
                "feddmc_pca_norm": pca_norm.detach().cpu(),
                "aggregation_weight": participant_weights.detach().cpu(),
            },
        )


__all__ = ["DMCDefense"]
