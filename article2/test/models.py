from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


def resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """ResNet18 for 32×32 RGB (e.g. CIFAR-10): 3×3 stem, no initial maxpool."""

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def resnet18_fashion_mnist(num_classes: int = 10) -> nn.Module:
    """ResNet18 for 32×32 grayscale (Fashion-MNIST resized): 1-channel stem, no maxpool."""

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Backward-compatible alias: same as ``resnet18_cifar10``."""

    return resnet18_cifar10(num_classes=num_classes)


class AGNewsClassifier(nn.Module):
    """Tiny Transformer text classifier with BN compatibility head.

    Transformer blocks mainly use LayerNorm. We keep a small BN head so the
    existing BN-feature-based SVDD pipeline can still extract features.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 4,
        padding_idx: int = 0,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.max_len = int(max_len)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) token ids, with PAD=0
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        pos = pos.clamp_max(self.max_len - 1)
        emb = self.embedding(x) + self.pos_embedding(pos)  # (B, L, E)
        pad_mask = x.eq(0)  # (B, L), True means masked
        h_seq = self.encoder(emb, src_key_padding_mask=pad_mask)  # (B, L, E)

        mask = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (h_seq * mask).sum(dim=1)  # (B, E)
        denom = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = summed / denom

        h = self.fc1(pooled)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.fc2(h)


def ag_news_classifier(
    vocab_size: int = 50000,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_classes: int = 4,
    padding_idx: int = 0,
    num_layers: int = 2,
    num_heads: int = 4,
    ff_dim: int = 256,
    max_len: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    return AGNewsClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        padding_idx=padding_idx,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_len=max_len,
        dropout=dropout,
    )


class Encoder(nn.Module):
    """Shallow encoder over BN feature vectors.

    All Linear layers use bias=False, final layer has no activation.
    """

    def __init__(self, d_bn: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_bn, 256, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Decoder that mirrors the encoder structure."""

    def __init__(self, d_bn: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, d_bn, bias=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class AutoEncoder(nn.Module):
    """AutoEncoder over BN features, providing reconstruction and encoding."""

    def __init__(self, d_bn: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(d_bn=d_bn, latent_dim=latent_dim)
        self.decoder = Decoder(d_bn=d_bn, latent_dim=latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct input features."""

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: Tensor) -> Tensor:
        """Encode input into latent representation."""

        return self.encoder(x)

