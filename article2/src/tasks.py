from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
_HF_DATASETS_IMPORT_ERROR = None
try:
    from datasets import load_dataset
except Exception as e:
    load_dataset = None
    _HF_DATASETS_IMPORT_ERROR = e

try:
    from .config import FedConfig
    from .models import ag_news_classifier, resnet18_cifar10, resnet18_fashion_mnist
except ImportError:
    from config import FedConfig
    from models import ag_news_classifier, resnet18_cifar10, resnet18_fashion_mnist


def _resolve_cifar10_root(config: FedConfig) -> str:
    """Root directory passed to torchvision CIFAR10.

    torchvision expects ``<root>/cifar-10-batches-py/``. Older layouts often put
    that folder directly under ``data_root``; newer code used ``data_root/cifar10``.
    Prefer whichever already exists so local data is found without re-download.
    """

    dr = os.path.normpath(config.data_root)
    flat = os.path.join(dr, "cifar-10-batches-py")
    nested = os.path.join(dr, "cifar10", "cifar-10-batches-py")
    if os.path.isdir(flat):
        return dr
    if os.path.isdir(nested):
        return os.path.join(dr, "cifar10")
    return dr


class FederatedTask(ABC):
    """One dataset + one backbone; plug in via TASK_REGISTRY."""

    name: str
    num_classes: int

    @abstractmethod
    def data_subdir(self, config: FedConfig) -> str:
        """Subfolder under config.data_root for this dataset."""

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Global model architecture for this task."""

    @abstractmethod
    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        """One train loader per client + one test loader (IID or Dirichlet label skew)."""

    def extract_svdd_features(self, config: FedConfig, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flattened vector for SVDD / AE (default: BatchNorm buffers + γ/β)."""

        try:
            from .utils import extract_bn_features
        except ImportError:
            from utils import extract_bn_features

        return extract_bn_features(state_dict)


class Cifar10Task(FederatedTask):
    name = "cifar10"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return _resolve_cifar10_root(config)

    def build_model(self) -> torch.nn.Module:
        return resnet18_cifar10(num_classes=self.num_classes)

    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor()])

        root = _resolve_cifar10_root(config)
        train_dataset: Dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        test_dataset: Dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(config, train_dataset, test_dataset, self.num_classes)


class FashionMnistTask(FederatedTask):
    name = "fashion_mnist"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "fashion_mnist")

    def build_model(self) -> torch.nn.Module:
        return resnet18_fashion_mnist(num_classes=self.num_classes)

    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        # Resize to 32×32 to match ResNet18 small-image stem (same spatial size as CIFAR-10).
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

        root = self.data_subdir(config)
        train_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform_train
        )
        test_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(config, train_dataset, test_dataset, self.num_classes)


class _TokenizedTextDataset(Dataset):
    """Map-style text dataset with integer labels and fixed-length token ids."""

    def __init__(
        self,
        rows: List[Tuple[int, str]],
        tokenizer,
        token_to_id: Dict[str, int],
        seq_len: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        self.seq_len = int(seq_len)
        self.pad_idx = int(token_to_id["<pad>"])
        self.unk_idx = int(token_to_id["<unk>"])
        # labels from HF AG News are already 0..3
        self.targets = [int(label) for label, _ in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str) -> torch.Tensor:
        toks = self.tokenizer(text)
        token_ids = [self.token_to_id.get(tok, self.unk_idx) for tok in toks]
        token_ids = token_ids[: self.seq_len]
        if len(token_ids) < self.seq_len:
            token_ids = token_ids + [self.pad_idx] * (self.seq_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label, text = self.rows[idx]
        x = self._encode(text)
        y = int(label)
        return x, y


def _basic_english_tokenize(text: str) -> List[str]:
    text = text.lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _build_token_vocab(rows: List[Tuple[int, str]], max_tokens: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for _, text in rows:
        for tok in _basic_english_tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keep_n = max(0, int(max_tokens) - 2)  # reserve <pad>, <unk>
    vocab_items = ["<pad>", "<unk>"] + [tok for tok, _ in sorted_tokens[:keep_n]]
    return {tok: idx for idx, tok in enumerate(vocab_items)}


def _normalize_proxy_env_schemes() -> Dict[str, str]:
    """Normalize proxy envs for httpx/huggingface compatibility.

    Some environments use `socks://host:port`, while httpx expects `socks5://`.
    """
    changed: Dict[str, str] = {}
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    for k in keys:
        v = os.environ.get(k)
        if not v:
            continue
        if v.lower().startswith("socks://"):
            changed[k] = v
            os.environ[k] = "socks5://" + v[len("socks://") :]
    return changed


class AGNewsTask(FederatedTask):
    name = "ag_news"
    num_classes = 4
    # Keep fixed architecture/ID-space so build_model() stays stateless.
    vocab_size = 50000
    seq_len = 128
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_heads = 4
    ff_dim = 256
    padding_idx = 0

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "ag_news")

    def build_model(self) -> torch.nn.Module:
        return ag_news_classifier(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            padding_idx=self.padding_idx,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            max_len=self.seq_len,
            dropout=0.1,
        )

    def build_dataloaders(self, config: FedConfig) -> Tuple[List[DataLoader], DataLoader]:
        if load_dataset is None:
            detail = (
                f" Root cause: {_HF_DATASETS_IMPORT_ERROR!r}"
                if _HF_DATASETS_IMPORT_ERROR is not None
                else ""
            )
            raise ImportError(
                "AG News task requires HuggingFace datasets."
                " Install with: pip install datasets" + detail
            )

        root = self.data_subdir(config)
        cache_dir = os.path.join(root, "hf_cache")
        _normalize_proxy_env_schemes()
        try:
            ds = load_dataset("ag_news", cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(
                "Failed to download/load AG News via HuggingFace datasets. "
                "If you use a proxy, ensure scheme is valid for httpx "
                "(e.g. socks5://127.0.0.1:7890, not socks://...). "
                f"Root cause: {e!r}"
            ) from e
        train_split = ds["train"]
        test_split = ds["test"]
        train_rows = [(int(item["label"]), str(item["text"])) for item in train_split]
        test_rows = [(int(item["label"]), str(item["text"])) for item in test_split]
        token_to_id = _build_token_vocab(train_rows, max_tokens=self.vocab_size)

        train_dataset: Dataset = _TokenizedTextDataset(
            rows=train_rows,
            tokenizer=_basic_english_tokenize,
            token_to_id=token_to_id,
            seq_len=self.seq_len,
        )
        test_dataset: Dataset = _TokenizedTextDataset(
            rows=test_rows,
            tokenizer=_basic_english_tokenize,
            token_to_id=token_to_id,
            seq_len=self.seq_len,
        )
        return _split_train_test_loaders(config, train_dataset, test_dataset, self.num_classes)

    def extract_svdd_features(self, config: FedConfig, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use Transformer LayerNorm (and optionally BN head) for SVDD — richer than BN-only."""

        try:
            from .utils import extract_ag_news_svdd_features
        except ImportError:
            from utils import extract_ag_news_svdd_features

        mode = getattr(config, "ag_news_svdd_features", "ln_bn")
        return extract_ag_news_svdd_features(state_dict, mode)


def _dataset_train_labels(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        t = dataset.targets
        if isinstance(t, list):
            return torch.tensor(t, dtype=torch.long)
        return torch.as_tensor(t, dtype=torch.long)
    raise TypeError(
        "Dirichlet / stratified split requires dataset.targets (e.g. torchvision CIFAR10/FashionMNIST)."
    )


def _client_index_lists_iid(num_samples: int, num_clients: int, seed: int) -> List[List[int]]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g).tolist()
    split_size = num_samples // num_clients
    out: List[List[int]] = []
    for i in range(num_clients):
        start = i * split_size
        end = num_samples if i == num_clients - 1 else (i + 1) * split_size
        out.append(perm[start:end])
    return out


def _fixed_client_quotas(num_samples: int, num_clients: int) -> List[int]:
    base = num_samples // num_clients
    rem = num_samples % num_clients
    return [base + (1 if i < rem else 0) for i in range(num_clients)]


def _client_index_lists_dirichlet_strict(
    labels: torch.Tensor,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Paper-style non-IID:
    1) sample client class priors q^(k) ~ Dir(alpha * p), p = uniform classes;
    2) allocate fixed sample quota per client using q^(k).
    """
    torch.manual_seed(seed)

    # Shuffle within each class to randomize picked images.
    class_pools: List[List[int]] = []
    for c in range(num_classes):
        idx_c = (labels == c).nonzero(as_tuple=True)[0]
        if idx_c.numel() == 0:
            class_pools.append([])
            continue
        perm = idx_c[torch.randperm(idx_c.numel())]
        class_pools.append(perm.tolist())

    # Prior p is uniform. Dirichlet concentration is alpha * p_i.
    p = torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float64)
    concentration = p * float(alpha)
    dist = torch.distributions.Dirichlet(concentration)
    q_by_client = dist.sample((num_clients,)).to(dtype=torch.float64)  # (K, C)

    quotas = _fixed_client_quotas(int(labels.numel()), num_clients)
    remaining = quotas[:]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # Keep assigning until all client quotas are filled.
    active_clients = [i for i in range(num_clients) if remaining[i] > 0]
    while active_clients:
        # Fill one sample at a time per active client to avoid starvation.
        for cid in list(active_clients):
            if remaining[cid] <= 0:
                continue

            probs = q_by_client[cid].clone()
            # Disable exhausted classes.
            for c in range(num_classes):
                if not class_pools[c]:
                    probs[c] = 0.0

            if float(probs.sum().item()) <= 0.0:
                # Fallback: pick any class that still has samples.
                available = [c for c in range(num_classes) if class_pools[c]]
                if not available:
                    raise RuntimeError("No samples left while client quota remains.")
                class_choice = available[torch.randint(0, len(available), (1,)).item()]
            else:
                probs = probs / probs.sum()
                class_choice = int(torch.multinomial(probs.float(), 1).item())

            client_indices[cid].append(class_pools[class_choice].pop())
            remaining[cid] -= 1

        active_clients = [i for i in active_clients if remaining[i] > 0]

    return client_indices


def _split_train_test_loaders(
    config: FedConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_classes: int,
) -> Tuple[List[DataLoader], DataLoader]:
    num_clients = config.num_clients
    alpha = config.dirichlet_alpha
    if alpha is None and config.dirichlet_noniid_beta is not None:
        alpha = config.dirichlet_noniid_beta

    if alpha is None:
        client_index_lists = _client_index_lists_iid(
            len(train_dataset), num_clients, config.seed
        )
    else:
        labels = _dataset_train_labels(train_dataset)
        client_index_lists = _client_index_lists_dirichlet_strict(
            labels, num_clients, num_classes, float(alpha), config.seed
        )

    client_loaders: List[DataLoader] = []
    for indices in client_index_lists:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(getattr(config, "num_workers", 0)),
            pin_memory=(config.device in ("cuda", "auto")),
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=(config.device in ("cuda", "auto")),
    )
    return client_loaders, test_loader


TASK_REGISTRY: Dict[str, Type[FederatedTask]] = {
    "cifar10": Cifar10Task,
    "fashion_mnist": FashionMnistTask,
    "ag_news": AGNewsTask,
}


def get_task(config: FedConfig) -> FederatedTask:
    key = config.task_name.lower().strip()
    cls = TASK_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown task_name: {config.task_name}. "
            f"Available: {sorted(TASK_REGISTRY.keys())}"
        )
    return cls()
