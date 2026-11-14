from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import Sample, load_mask, load_rgb
from .metrics import binary_metrics, pixel_metrics
from .models import (
    ConvAutoEncoder,
    FeatureAnomalyModel,
    FeatureAutoEncoder,
    fit_standardizer,
)
from .patches import assemble, extract_patches, top_fraction_mean


@dataclass
class FeatureConfig:
    backbone: str = "resnet18"
    patch_size: int = 64
    stride: int = 32
    upscale: bool = False
    hidden_dim: int = 256
    latent_dim: int = 64
    dropout: float = 0.0
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 60
    patience: int = 12

    @property
    def name(self) -> str:
        return (
            f"fae_{self.backbone}_p{self.patch_size}_s{self.stride}_u{int(self.upscale)}"
            f"_h{self.hidden_dim}_l{self.latent_dim}_d{self.dropout}_wd{self.weight_decay}"
        )


@dataclass
class CAEConfig:
    patch_size: int = 64
    stride: int = 32
    latent_channels: int = 32
    base_channels: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 60
    patience: int = 12

    @property
    def name(self) -> str:
        return f"cae_p{self.patch_size}_s{self.stride}_lc{self.latent_channels}_bc{self.base_channels}_wd{self.weight_decay}"


def _feature_tensor(cache: dict[str, np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(cache["features"], dtype=np.float32))


def _evaluate_reconstruction(
    model: FeatureAnomalyModel,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            normalized = (batch - model.feature_mean) / model.feature_std
            reconstruction = model.autoencoder(normalized)
            losses.append(float(torch.mean((normalized - reconstruction) ** 2)))
    return float(np.mean(losses))


def train_feature_ae(
    train_cache: dict[str, np.ndarray],
    val_cache: dict[str, np.ndarray],
    config: FeatureConfig,
    device: str = "cuda",
) -> tuple[FeatureAnomalyModel, list[dict[str, float]]]:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(device)
    train_features = _feature_tensor(train_cache)
    val_features = _feature_tensor(val_cache)
    mean, std = fit_standardizer(train_features)
    autoencoder = FeatureAutoEncoder(
        input_dim=train_features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)
    model = FeatureAnomalyModel(autoencoder, mean, std).to(device)

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    loader = DataLoader(
        TensorDataset(train_features),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            normalized = (batch - model.feature_mean) / model.feature_std
            reconstruction = autoencoder(normalized)
            loss = torch.mean((normalized - reconstruction) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        scheduler.step()
        train_loss = float(np.mean(batch_losses))
        val_loss = _evaluate_reconstruction(
            model, val_features, device, config.batch_size
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_feature_maps(
    model: FeatureAnomalyModel,
    cache: dict[str, np.ndarray],
    samples: list[Sample],
    patch_size: int,
    image_size: int,
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    device = torch.device(device)
    features = _feature_tensor(cache)
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            scores.append(model(features[start : start + batch_size].to(device)).cpu())
    patch_scores = torch.cat(scores).numpy()
    image_index = np.asarray(cache["image_index"])
    coords = np.asarray(cache["coords"])
    maps = []
    image_scores = []
    masks = []
    for index, sample in enumerate(samples):
        mask_patches = image_index == index
        score_map = assemble(
            patch_scores[mask_patches],
            [tuple(c) for c in coords[mask_patches]],
            image_size,
            patch_size,
        )
        maps.append(score_map)
        image_scores.append(top_fraction_mean(score_map))
        masks.append(
            load_mask(sample.mask_path, image_size).squeeze(0).numpy().astype(np.uint8)
        )
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    return (
        np.asarray(image_scores),
        labels,
        maps,
        masks,
        np.asarray([s.defect for s in samples], dtype=object),
    )


def evaluate_feature_model(
    model: FeatureAnomalyModel,
    cache: dict[str, np.ndarray],
    samples: list[Sample],
    patch_size: int,
    image_size: int,
    device: str = "cuda",
) -> dict[str, float]:
    scores, labels, maps, masks, defects = predict_feature_maps(
        model, cache, samples, patch_size, image_size, device
    )
    image_level = binary_metrics(scores, labels)
    pixel_level = pixel_metrics(maps, masks)
    per_defect = {}
    for defect in sorted(set(defects[labels == 1])):
        selected = defects == defect
        per_defect[defect] = float(scores[selected].mean())
    return {**image_level, **pixel_level, "per_defect_mean_score": per_defect}


def samples_to_patches(
    samples: list[Sample], patch_size: int, stride: int, image_size: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    patches = []
    image_index = []
    coords = []
    for index, sample in enumerate(samples):
        image = load_rgb(sample.image_path, image_size)
        values, positions = extract_patches(image, patch_size, stride)
        patches.append((values * 255).to(torch.uint8))
        image_index.extend([index] * len(values))
        coords.extend(positions)
    return (
        torch.cat(patches),
        np.asarray(image_index, dtype=np.int16),
        np.asarray(coords, dtype=np.int16),
    )


def train_cae(
    train_cache: tuple[torch.Tensor, np.ndarray, np.ndarray],
    val_cache: tuple[torch.Tensor, np.ndarray, np.ndarray],
    config: CAEConfig,
    device: str = "cuda",
) -> tuple[ConvAutoEncoder, list[dict[str, float]]]:
    torch.manual_seed(0)
    device = torch.device(device)
    train_patches = train_cache[0]
    val_patches = val_cache[0]
    model = ConvAutoEncoder(config.latent_channels, config.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    loader = DataLoader(
        TensorDataset(train_patches), batch_size=config.batch_size, shuffle=True
    )

    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device).float() / 255.0
            reconstruction = model(batch)
            loss = torch.mean((batch - reconstruction) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        train_loss = float(np.mean(losses))
        model.eval()
        with torch.inference_mode():
            val_losses = []
            for start in range(0, len(val_patches), config.batch_size):
                batch = (
                    val_patches[start : start + config.batch_size].to(device).float()
                    / 255.0
                )
                reconstruction = model(batch)
                val_losses.append(float(torch.mean((batch - reconstruction) ** 2)))
        val_loss = float(np.mean(val_losses))
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_cae_maps(
    model: ConvAutoEncoder,
    samples: list[Sample],
    patch_size: int,
    stride: int,
    image_size: int,
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    device = torch.device(device)
    model.eval()
    maps = []
    image_scores = []
    masks = []
    with torch.inference_mode():
        for sample in samples:
            image = load_rgb(sample.image_path, image_size)
            patches, coords = extract_patches(image, patch_size, stride)
            outputs = []
            for start in range(0, len(patches), batch_size):
                batch = patches[start : start + batch_size].to(device)
                outputs.append(model.pixel_scores(batch).cpu())
            pixel_scores = torch.cat(outputs).numpy()
            score_map = assemble(pixel_scores, coords, image_size, patch_size)
            maps.append(score_map)
            image_scores.append(top_fraction_mean(score_map))
            masks.append(
                load_mask(sample.mask_path, image_size)
                .squeeze(0)
                .numpy()
                .astype(np.uint8)
            )
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    return (
        np.asarray(image_scores),
        labels,
        maps,
        masks,
        np.asarray([s.defect for s in samples], dtype=object),
    )


def evaluate_cae_model(
    model: ConvAutoEncoder,
    samples: list[Sample],
    patch_size: int,
    stride: int,
    image_size: int,
    device: str = "cuda",
) -> dict[str, float]:
    scores, labels, maps, masks, defects = predict_cae_maps(
        model, samples, patch_size, stride, image_size, device
    )
    image_level = binary_metrics(scores, labels)
    pixel_level = pixel_metrics(maps, masks)
    per_defect = {
        defect: float(scores[defects == defect].mean())
        for defect in sorted(set(defects[labels == 1]))
    }
    return {**image_level, **pixel_level, "per_defect_mean_score": per_defect}


def time_prediction(function, repeats: int = 3) -> float:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))
