from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .data import Sample, load_rgb
from .patches import extract_patches


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    output_dim: int
    weights: object


BACKBONES = {
    "resnet18": BackboneSpec("resnet18", 512, ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": BackboneSpec(
        "mobilenet_v3_small", 576, MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ),
}


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Identity()
        return model, 512
    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier = nn.Identity()
        return model, 576
    raise ValueError(f"unknown backbone: {name}")


class PatchFeatureExtractor:
    def __init__(
        self,
        backbone: str = "resnet18",
        patch_size: int = 64,
        stride: int = 32,
        upscale: bool = False,
        image_size: int = IMAGE_SIZE,
        device: str | torch.device = "cuda",
        batch_size: int = 128,
    ):
        self.backbone_name = backbone
        self.patch_size = patch_size
        self.stride = stride
        self.upscale = upscale
        self.image_size = image_size
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, self.output_dim = build_backbone(backbone, pretrained=True)
        self.model.to(self.device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract_patches(self, patches: torch.Tensor) -> torch.Tensor:
        patches = patches.to(self.device)
        if self.upscale and self.patch_size != 224:
            patches = torch.nn.functional.interpolate(
                patches, size=(224, 224), mode="bilinear", align_corners=False
            )
        patches = (patches - self.mean) / self.std
        outputs = []
        for start in range(0, len(patches), self.batch_size):
            batch = patches[start : start + self.batch_size]
            outputs.append(self.model(batch).flatten(1))
        return torch.cat(outputs).float().cpu()

    def extract_sample(
        self, sample: Sample
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        image = load_rgb(sample.image_path, self.image_size)
        patches, coords = extract_patches(image, self.patch_size, self.stride)
        features = self.extract_patches(patches).numpy()
        return features, coords

    def extract_samples(self, samples: list[Sample]) -> dict[str, np.ndarray]:
        all_features = []
        all_coords = []
        image_index = []
        for index, sample in enumerate(samples):
            features, coords = self.extract_sample(sample)
            all_features.append(features)
            all_coords.extend(coords)
            image_index.extend([index] * len(coords))
        return {
            "features": np.concatenate(all_features).astype(np.float32),
            "coords": np.asarray(all_coords, dtype=np.int16),
            "image_index": np.asarray(image_index, dtype=np.int16),
            "paths": np.asarray([str(s.image_path) for s in samples]),
            "labels": np.asarray([s.label for s in samples], dtype=np.int8),
            "defects": np.asarray([s.defect for s in samples]),
        }

    def cache_path(self, cache_dir: Path, category: str, split: str) -> Path:
        tag = f"{category}_{self.backbone_name}_p{self.patch_size}_s{self.stride}_u{int(self.upscale)}_{split}"
        return cache_dir / f"{tag}.npz"

    def extract_to_cache(
        self,
        cache_dir: Path,
        category: str,
        split: str,
        samples: list[Sample],
        overwrite: bool = False,
    ) -> dict[str, np.ndarray]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(cache_dir, category, split)
        if path.exists() and not overwrite:
            return dict(np.load(path, allow_pickle=True))
        data = self.extract_samples(samples)
        np.savez_compressed(path, **data)
        return data
