from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Create ResNet18 backbone for CIFAR-10."""

    model = resnet18(weights=None, num_classes=num_classes)
    return model


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

