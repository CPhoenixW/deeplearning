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

