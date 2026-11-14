from __future__ import annotations

import torch
from torch import nn


class FeatureAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


class FeatureAnomalyModel(nn.Module):
    def __init__(
        self,
        autoencoder: FeatureAutoEncoder,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        reconstruction = self.autoencoder(normalized)
        return ((normalized - reconstruction) ** 2).mean(dim=1)

    def reconstruct(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        reconstruction = self.autoencoder(normalized)
        return normalized, reconstruction

    @property
    def input_dim(self) -> int:
        return int(self.feature_mean.numel())


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    return mean, std


class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_channels: int = 32, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, latent_channels, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, c * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(patches))

    def pixel_scores(self, patches: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(patches)
        return ((patches - reconstruction) ** 2).mean(dim=1)
