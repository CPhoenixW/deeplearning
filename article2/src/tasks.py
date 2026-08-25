from __future__ import annotations

import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
_HF_DATASETS_IMPORT_ERROR = None
try:
    from datasets import config as hf_datasets_config
    from datasets import load_dataset
except Exception as e:
    hf_datasets_config = None
    load_dataset = None
    _HF_DATASETS_IMPORT_ERROR = e

try:
    from .config import FedConfig
    from .models import (
        ag_news_classifier,
        covid19_resnet50,
        fashion_mnist_cnn,
        lenet_grayscale,
        resnet18_cifar10,
    )
except ImportError:
    from config import FedConfig
    from models import (
        ag_news_classifier,
        covid19_resnet50,
        fashion_mnist_cnn,
        lenet_grayscale,
        resnet18_cifar10,
    )


SERVER_VALIDATION_BATCH_SIZE = 64


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
    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        """Client loaders, fixed clean validation loader, and test loader."""

class Cifar10Task(FederatedTask):
    name = "cifar10"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return _resolve_cifar10_root(config)

    def build_model(self) -> torch.nn.Module:
        return resnet18_cifar10(num_classes=self.num_classes)

    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
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
        validation_dataset: Dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_test
        )
        test_dataset: Dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(
            config, train_dataset, validation_dataset, test_dataset, self.num_classes
        )


class FashionMnistTask(FederatedTask):
    name = "fashion_mnist"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "fashion_mnist")

    def build_model(self) -> torch.nn.Module:
        return fashion_mnist_cnn(num_classes=self.num_classes)

    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

        root = self.data_subdir(config)
        train_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform_train
        )
        validation_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform_test
        )
        test_dataset: Dataset = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform_test
        )
        return _split_train_test_loaders(
            config, train_dataset, validation_dataset, test_dataset, self.num_classes
        )


class MnistTask(FederatedTask):
    name = "mnist"
    num_classes = 10

    def data_subdir(self, config: FedConfig) -> str:
        return os.path.join(config.data_root, "mnist")

    def build_model(self) -> torch.nn.Module:
        return lenet_grayscale(num_classes=self.num_classes)

    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        root = self.data_subdir(config)
        train_dataset: Dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        validation_dataset: Dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform_test)
        test_dataset: Dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
        return _split_train_test_loaders(
            config, train_dataset, validation_dataset, test_dataset, self.num_classes
        )


COVID19_CLASS_TO_INDEX: Dict[str, int] = {
    "COVID": 0,
    "Lung_Opacity": 1,
    "Normal": 2,
    "Viral_Pneumonia": 3,
}
COVID19_SPLIT_SEED = 42

_COVID19_SOURCE_DIRS: Dict[str, Tuple[str, ...]] = {
    "COVID": ("COVID",),
    "Lung_Opacity": ("Lung_Opacity", "Lung Opacity"),
    "Normal": ("Normal",),
    "Viral_Pneumonia": ("Viral Pneumonia", "Viral_Pneumonia"),
}

_COVID19_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _has_covid19_class_dirs(root: Path) -> bool:
    return all(
        any((root / directory).is_dir() for directory in source_dirs)
        for source_dirs in _COVID19_SOURCE_DIRS.values()
    )


def _resolve_covid19_root(config: FedConfig) -> str:
    data_root = Path(config.data_root).expanduser()
    candidates = (
        data_root / "covid19" / "COVID-19_Radiography_Dataset",
        data_root / "COVID-19_Radiography_Dataset",
        data_root / "covid19",
    )
    for candidate in candidates:
        if _has_covid19_class_dirs(candidate):
            return str(candidate)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "COVID-19 Radiography Database was not found. Expected all four class "
        f"directories under one of: {searched}"
    )


def scan_covid19_records(root: str | Path) -> List[Tuple[str, int]]:
    """Return image paths with the protocol's fixed class-to-index mapping."""

    dataset_root = Path(root)
    if not _has_covid19_class_dirs(dataset_root):
        raise FileNotFoundError(
            f"Missing COVID-19 Radiography class directories under {dataset_root}"
        )
    records: List[Tuple[str, int]] = []
    for class_name, label in COVID19_CLASS_TO_INDEX.items():
        aliases = _COVID19_SOURCE_DIRS[class_name]
        class_dir = next(
            dataset_root / alias
            for alias in aliases
            if (dataset_root / alias).is_dir()
        )
        image_root = class_dir / "images" if (class_dir / "images").is_dir() else class_dir
        paths = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _COVID19_IMAGE_SUFFIXES
            and "masks" not in {part.lower() for part in path.parts}
        )
        if not paths:
            raise FileNotFoundError(
                f"No radiographs found for class {class_name!r} in {image_root}"
            )
        records.extend((str(path), label) for path in paths)
    return records


def _stratified_train_test_indices(
    labels: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    train_fraction: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """Build deterministic per-class train/test indices without rebalancing."""

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(int(seed) + 130363)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for cls in range(int(num_classes)):
        available = torch.where(labels == cls)[0]
        if int(available.numel()) < 2:
            raise ValueError(f"Class {cls} needs at least two images for an 80/20 split")
        order = available[torch.randperm(int(available.numel()), generator=generator)]
        train_count = int(float(train_fraction) * int(available.numel()))
        train_count = min(max(train_count, 1), int(available.numel()) - 1)
        train_indices.extend(int(index) for index in order[:train_count].tolist())
        test_indices.extend(int(index) for index in order[train_count:].tolist())
    train_order = torch.randperm(len(train_indices), generator=generator).tolist()
    test_order = torch.randperm(len(test_indices), generator=generator).tolist()
    return (
        [train_indices[index] for index in train_order],
        [test_indices[index] for index in test_order],
    )


class Covid19RadiographyDataset(Dataset):
    """Map-style X-ray dataset backed by explicit ``(path, label)`` records."""

    def __init__(self, records: List[Tuple[str, int]], transform) -> None:
        self.records = list(records)
        self.targets = [int(label) for _path, label in self.records]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert("L")
        return self.transform(image), int(label)


class Covid19Task(FederatedTask):
    name = "covid19"
    num_classes = 4

    def data_subdir(self, config: FedConfig) -> str:
        return _resolve_covid19_root(config)

    def build_model(self) -> torch.nn.Module:
        return covid19_resnet50(num_classes=self.num_classes)

    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        records = scan_covid19_records(self.data_subdir(config))
        labels = torch.tensor([label for _path, label in records], dtype=torch.long)
        train_indices, test_indices = _stratified_train_test_indices(
            labels,
            num_classes=self.num_classes,
            seed=COVID19_SPLIT_SEED,
            train_fraction=0.8,
        )
        train_records = [records[index] for index in train_indices]
        test_records = [records[index] for index in test_indices]
        train_dataset: Dataset = Covid19RadiographyDataset(
            train_records, transform_train
        )
        validation_dataset: Dataset = Covid19RadiographyDataset(
            train_records, transform_test
        )
        test_dataset: Dataset = Covid19RadiographyDataset(
            test_records, transform_test
        )
        return _split_train_test_loaders(
            config, train_dataset, validation_dataset, test_dataset, self.num_classes
        )


class _TokenizedTextDataset(Dataset):
    """Map-style AG News dataset with cached, fixed-length token ids.

    Tokenizing inside ``__getitem__`` is needlessly expensive for federated
    training because the same examples are visited once per client and once
    per communication round.  Encode the complete split once during dataset
    construction so training only performs a tensor lookup.
    """

    def __init__(
        self,
        rows: List[Tuple[int, str]],
        tokenizer,
        token_to_id: Dict[str, int],
        seq_len: int,
    ) -> None:
        self.seq_len = int(seq_len)
        self.pad_idx = int(token_to_id["<pad>"])
        self.unk_idx = int(token_to_id["<unk>"])

        # Labels from HF AG News are already 0..3.  Keep the encoded examples
        # as one contiguous tensor so __getitem__ does no Python tokenization
        # or per-sample tensor construction during training.
        self.targets = torch.empty(len(rows), dtype=torch.long)
        self.input_ids = torch.full(
            (len(rows), self.seq_len), self.pad_idx, dtype=torch.long
        )
        for row_idx, (label, text) in enumerate(rows):
            self.targets[row_idx] = int(label)
            tokens = tokenizer(text)
            token_ids = [
                token_to_id.get(token, self.unk_idx)
                for token in tokens[: self.seq_len]
            ]
            if token_ids:
                self.input_ids[row_idx, : len(token_ids)] = torch.as_tensor(
                    token_ids, dtype=torch.long
                )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.input_ids[idx], self.targets[idx]


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

    def build_dataloaders(
        self, config: FedConfig
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
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
        offline = bool(config.hf_datasets_offline)
        os.environ["HF_DATASETS_OFFLINE"] = "1" if offline else "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
        if hf_datasets_config is not None:
            hf_datasets_config.HF_DATASETS_OFFLINE = offline
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
        return _split_train_test_loaders(
            config, train_dataset, train_dataset, test_dataset, self.num_classes
        )

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
    validation_dataset: Dataset,
    test_dataset: Dataset,
    num_classes: int,
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    labels = _dataset_train_labels(train_dataset)
    validation_indices = _validation_indices(
        labels,
        num_classes=num_classes,
        seed=int(config.seed),
        size=int(config.server_validation_size),
    )
    validation_set = Subset(validation_dataset, validation_indices)
    validation_loader = DataLoader(
        validation_set,
        batch_size=min(SERVER_VALIDATION_BATCH_SIZE, len(validation_set)),
        shuffle=False,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=(config.device in ("cuda", "auto")),
    )
    validation_lookup = set(validation_indices)
    available_indices = [
        index for index in range(len(train_dataset)) if index not in validation_lookup
    ]
    available_labels = labels[available_indices]
    class_weight_mode = str(getattr(config, "class_weight_mode", "none")).lower().strip()
    counts = torch.bincount(available_labels, minlength=int(num_classes)).float()
    if bool((counts <= 0).any().item()):
        raise ValueError("Every class needs at least one client-training sample.")
    inverse_weights = available_labels.numel() / (float(num_classes) * counts)
    inverse_weights = inverse_weights / inverse_weights.mean()
    if class_weight_mode == "none":
        config.client_class_weights = None
    elif class_weight_mode == "inverse_frequency":
        config.client_class_weights = [float(value) for value in inverse_weights.tolist()]
    else:
        raise ValueError(
            f"Unknown class_weight_mode {config.class_weight_mode!r}; "
            "use 'none' or 'inverse_frequency'."
        )
    num_clients = config.num_clients
    alpha = config.dirichlet_alpha
    if alpha is None and config.dirichlet_noniid_beta is not None:
        alpha = config.dirichlet_noniid_beta

    if alpha is None:
        relative_lists = _client_index_lists_iid(
            len(available_indices), num_clients, config.seed
        )
    else:
        relative_lists = _client_index_lists_dirichlet_strict(
            available_labels, num_clients, num_classes, float(alpha), config.seed
        )
    client_index_lists = [
        [available_indices[index] for index in indices]
        for indices in relative_lists
    ]

    client_loaders: List[DataLoader] = []
    sampling_mode = str(
        getattr(config, "client_sampling_mode", "none")
    ).lower().strip()
    if sampling_mode not in {"none", "balanced"}:
        raise ValueError(
            f"Unknown client_sampling_mode {config.client_sampling_mode!r}; "
            "use 'none' or 'balanced'."
        )
    for indices in client_index_lists:
        subset = Subset(train_dataset, indices)
        sampler = None
        if sampling_mode == "balanced":
            subset_labels = labels[indices]
            subset_counts = torch.bincount(
                subset_labels, minlength=int(num_classes)
            ).float()
            present = subset_counts > 0
            subset_weights = torch.zeros_like(subset_counts)
            subset_weights[present] = 1.0 / subset_counts[present]
            subset_weights[present] = (
                subset_weights[present] / subset_weights[present].mean()
            )
            sample_weights = subset_weights[subset_labels].double()
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(indices),
                replacement=True,
            )
        loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
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
    return client_loaders, validation_loader, test_loader


def _validation_indices(
    labels: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    size: int,
) -> List[int]:
    """Select exactly ``size`` deterministic, class-balanced clean samples."""

    total = int(size)
    if total < 1:
        raise ValueError("server_validation_size must be at least 1.")
    if total > int(labels.numel()):
        raise ValueError(
            "server_validation_size cannot exceed the available training samples."
        )
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    base = total // int(num_classes)
    remainder = total % int(num_classes)
    selected: list[int] = []
    for cls in range(int(num_classes)):
        available = torch.where(labels == cls)[0]
        count = base + (1 if cls < remainder else 0)
        if count > int(available.numel()):
            raise ValueError(f"Not enough class-{cls} samples for server validation.")
        order = torch.randperm(int(available.numel()), generator=generator)[:count]
        selected.extend(int(item) for item in available[order].tolist())
    shuffle = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[index] for index in shuffle]


TASK_REGISTRY: Dict[str, Type[FederatedTask]] = {
    "cifar10": Cifar10Task,
    "covid19": Covid19Task,
    "mnist": MnistTask,
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
