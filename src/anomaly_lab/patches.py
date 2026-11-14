from __future__ import annotations

import numpy as np
import torch


def patch_coords(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > length:
        raise ValueError(f"patch size {patch_size} exceeds image size {length}")
    coords = list(range(0, length - patch_size + 1, stride))
    if coords[-1] != length - patch_size:
        coords.append(length - patch_size)
    return coords


def grid_coords(
    height: int, width: int, patch_size: int, stride: int
) -> list[tuple[int, int]]:
    return [
        (y, x)
        for y in patch_coords(height, patch_size, stride)
        for x in patch_coords(width, patch_size, stride)
    ]


def extract_patches(
    image: torch.Tensor, patch_size: int, stride: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    _, height, width = image.shape
    coords = grid_coords(height, width, patch_size, stride)
    patches = torch.empty(
        len(coords), image.shape[0], patch_size, patch_size, dtype=image.dtype
    )
    for index, (y, x) in enumerate(coords):
        patches[index] = image[:, y : y + patch_size, x : x + patch_size]
    return patches, coords


def assemble(
    values: np.ndarray | torch.Tensor,
    coords: list[tuple[int, int]],
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    if array.ndim == 1:
        array = array[:, None, None] * np.ones(
            (len(coords), patch_size, patch_size), dtype=np.float32
        )
    total = np.zeros((image_size, image_size), dtype=np.float32)
    count = np.zeros((image_size, image_size), dtype=np.float32)
    for index, (y, x) in enumerate(coords):
        total[y : y + patch_size, x : x + patch_size] += array[index]
        count[y : y + patch_size, x : x + patch_size] += 1.0
    return total / np.maximum(count, 1.0)


def top_fraction_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.sort(values.reshape(-1))[::-1]
    k = max(1, round(flat.size * fraction))
    return float(flat[:k].mean())


def gaussian_smooth(array: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(array, sigma=sigma, mode="nearest")
