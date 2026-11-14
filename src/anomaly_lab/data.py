from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGE_SIZE, MVTEC_DIR


@dataclass(frozen=True)
class Sample:
    category: str
    defect: str
    label: int
    image_path: Path
    mask_path: Path | None = None


def list_mvtec_samples(root: Path, category: str, split: str) -> list[Sample]:
    category_root = root / category
    if split == "train":
        files = sorted((category_root / "train" / "good").glob("*.png"))
        return [Sample(category, "good", 0, path) for path in files]

    samples: list[Sample] = []
    test_root = category_root / "test"
    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect = defect_dir.name
        for path in sorted(defect_dir.glob("*.png")):
            if defect == "good":
                samples.append(Sample(category, defect, 0, path))
                continue
            mask_path = (
                category_root / "ground_truth" / defect / f"{path.stem}_mask.png"
            )
            samples.append(Sample(category, defect, 1, path, mask_path))
    return samples


def split_train_normal(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    n_val = max(1, round(len(samples) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def load_rgb(path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_mask(path: Path | None, size: int = IMAGE_SIZE) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, size, size)
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32))[None]


class MvtecDataset(Dataset):
    def __init__(self, root: Path, samples: list[Sample], image_size: int = IMAGE_SIZE):
        self.root = root
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_rgb(sample.image_path, self.image_size)
        mask = load_mask(sample.mask_path, self.image_size)
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "defect": sample.defect,
            "path": str(sample.image_path),
        }


def default_root(root: Path | None = None) -> Path:
    return root if root is not None else MVTEC_DIR
