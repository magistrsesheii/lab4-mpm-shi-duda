from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import CASTING_DIR, MVTEC_DIR, PCB_DIR, TEXTURE_CATEGORIES
from .data import Sample, list_mvtec_samples
from .features import PatchFeatureExtractor
from .metrics import binary_metrics
from .patches import assemble, top_fraction_mean


def _image_sizes(paths: list[Path]) -> tuple[float, float, float, float]:
    widths = []
    heights = []
    for path in paths:
        with Image.open(path) as image:
            widths.append(image.width)
            heights.append(image.height)
    return (
        float(np.mean(widths)),
        float(np.mean(heights)),
        float(np.std(widths)),
        float(np.std(heights)),
    )


def _mask_fractions(masks: list[np.ndarray]) -> tuple[float, float, float]:
    fractions = np.asarray([float((mask > 0.5).mean()) for mask in masks])
    return float(fractions.mean()), float(fractions.std()), float(fractions.max())


def mvtec_stats(root: Path, category: str) -> dict[str, object]:
    train = list_mvtec_samples(root, category, "train")
    test = list_mvtec_samples(root, category, "test")
    defects = [sample for sample in test if sample.label == 1]
    defect_types = sorted({sample.defect for sample in defects})
    counts = {
        name: sum(sample.defect == name for sample in defects) for name in defect_types
    }
    masks = []
    for sample in defects:
        with Image.open(sample.mask_path).convert("L") as mask:
            masks.append(np.asarray(mask) > 127)
    mean_area, std_area, max_area = _mask_fractions(masks)
    width, height, width_std, height_std = _image_sizes([s.image_path for s in test])
    return {
        "dataset": "mvtec",
        "category": category,
        "family": "texture" if category in TEXTURE_CATEGORIES else "object",
        "train_normal": len(train),
        "test_normal": sum(sample.label == 0 for sample in test),
        "test_anomaly": len(defects),
        "anomaly_fraction": len(defects) / len(test),
        "defect_types": len(defect_types),
        "smallest_defect_type": min(counts.values()) if counts else 0,
        "largest_defect_type": max(counts.values()) if counts else 0,
        "defect_imbalance": max(counts.values()) / max(1, min(counts.values()))
        if counts
        else 1.0,
        "mean_mask_area": mean_area,
        "std_mask_area": std_area,
        "max_mask_area": max_area,
        "width": width,
        "height": height,
        "width_std": width_std,
        "height_std": height_std,
        "pixel_masks": True,
    }


def casting_stats(root: Path) -> dict[str, object]:
    train_ok = sorted(
        (root / "casting_data" / "casting_data" / "train" / "ok_front").glob("*.jpeg")
    )
    train_def = sorted(
        (root / "casting_data" / "casting_data" / "train" / "def_front").glob("*.jpeg")
    )
    test_ok = sorted(
        (root / "casting_data" / "casting_data" / "test" / "ok_front").glob("*.jpeg")
    )
    test_def = sorted(
        (root / "casting_data" / "casting_data" / "test" / "def_front").glob("*.jpeg")
    )
    width, height, width_std, height_std = _image_sizes(test_ok + test_def)
    anomaly_fraction = len(test_def) / max(1, len(test_ok) + len(test_def))
    return {
        "dataset": "casting",
        "category": "casting",
        "family": "object",
        "train_normal": len(train_ok),
        "test_normal": len(test_ok),
        "test_anomaly": len(test_def),
        "anomaly_fraction": anomaly_fraction,
        "defect_types": 1,
        "smallest_defect_type": len(train_def) + len(test_def),
        "largest_defect_type": len(train_def) + len(test_def),
        "defect_imbalance": 1.0,
        "mean_mask_area": float("nan"),
        "std_mask_area": float("nan"),
        "max_mask_area": float("nan"),
        "width": width,
        "height": height,
        "width_std": width_std,
        "height_std": height_std,
        "pixel_masks": False,
    }


def _pcb_defect_counts(annotation_dir: Path) -> dict[str, int]:
    import xml.etree.ElementTree as ET

    counts: dict[str, int] = {}
    for path in sorted(annotation_dir.glob("*.xml")):
        names = {
            obj.findtext("name") for obj in ET.parse(path).getroot().findall("object")
        }
        for name in names:
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def pcb_stats(root: Path) -> dict[str, object]:
    split = "test_data" if (root / "test_data").exists() else "train_data"
    test_images = sorted((root / split / "JPEGImages").glob("*.jpeg"))
    counts = _pcb_defect_counts(root / split / "Annotations")
    width, height, width_std, height_std = _image_sizes(test_images)
    return {
        "dataset": "pcb_aoi",
        "category": "pcb_aoi",
        "family": "object",
        "train_normal": 0,
        "test_normal": 0,
        "test_anomaly": len(test_images),
        "anomaly_fraction": 1.0,
        "defect_types": len(counts),
        "smallest_defect_type": min(counts.values()) if counts else 0,
        "largest_defect_type": max(counts.values()) if counts else 0,
        "defect_imbalance": max(counts.values()) / max(1, min(counts.values()))
        if counts
        else 1.0,
        "mean_mask_area": float("nan"),
        "std_mask_area": float("nan"),
        "max_mask_area": float("nan"),
        "width": width,
        "height": height,
        "width_std": width_std,
        "height_std": height_std,
        "pixel_masks": False,
    }


def metadata_table(
    mvtec_root: Path = MVTEC_DIR,
    casting_root: Path = CASTING_DIR,
    pcb_root: Path = PCB_DIR,
):
    rows = []
    for category in sorted(path.name for path in mvtec_root.iterdir() if path.is_dir()):
        rows.append(mvtec_stats(mvtec_root, category))
    rows.append(casting_stats(casting_root))
    rows.append(pcb_stats(pcb_root))
    return rows


def _subsample(samples: list[Sample], limit: int, seed: int) -> list[Sample]:
    if len(samples) <= limit:
        return samples
    rng = np.random.default_rng(seed)
    index = rng.choice(len(samples), size=limit, replace=False)
    return [samples[i] for i in index]


def knn_image_scores(
    extractor: PatchFeatureExtractor,
    memory_samples: list[Sample],
    query_samples: list[Sample],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    memory = extractor.extract_samples(memory_samples)
    query = extractor.extract_samples(query_samples)
    memory_features = memory["features"]
    memory_features = memory_features / np.linalg.norm(
        memory_features, axis=1, keepdims=True
    )
    query_features = query["features"]
    query_features = query_features / np.linalg.norm(
        query_features, axis=1, keepdims=True
    )

    distances = 1.0 - query_features @ memory_features.T
    nearest = distances.min(axis=1)

    scores = []
    maps = []
    masks = []
    for index in range(len(query_samples)):
        mask = query["image_index"] == index
        patch_scores = nearest[mask]
        score_map = assemble(
            patch_scores,
            [tuple(c) for c in query["coords"][mask]],
            image_size,
            extractor.patch_size,
        )
        scores.append(top_fraction_mean(score_map))
        maps.append(score_map)
        masks.append(np.zeros_like(score_map))
    return (
        np.asarray(scores),
        query["labels"],
        np.asarray(
            [query["defects"][i] for i in range(len(query_samples))], dtype=object
        ),
        maps,
        masks,
    )


def screen_candidates(
    mvtec_root: Path,
    casting_root: Path,
    categories: list[str],
    train_limit: int = 40,
    test_limit: int = 80,
    seed: int = 0,
    device: str = "cuda",
) -> list[dict[str, object]]:
    extractor = PatchFeatureExtractor(
        backbone="resnet18", patch_size=64, stride=32, image_size=256, device=device
    )
    rows = []
    for category in categories:
        train = list_mvtec_samples(mvtec_root, category, "train")
        test = list_mvtec_samples(mvtec_root, category, "test")
        memory = _subsample(train, train_limit, seed)
        positives = [s for s in test if s.label == 1]
        negatives = [s for s in test if s.label == 0]
        positives = _subsample(positives, test_limit // 2, seed + 1)
        negatives = _subsample(negatives, test_limit // 2, seed + 2)
        query = sorted(positives + negatives, key=lambda item: item.image_path.name)
        scores, labels, _, _, _ = knn_image_scores(extractor, memory, query, 256)
        metrics = binary_metrics(scores, labels)
        rows.append(
            {
                "candidate": category,
                "kind": "mvtec",
                "n_memory": len(memory),
                "n_query": len(query),
                "knn_auroc": metrics["auroc"],
                "knn_aupr": metrics["aupr"],
                "knn_f1": metrics["f1"],
            }
        )
        print(f"{category:12s} knn_auroc={metrics['auroc']:.4f}")

    casting_train = sorted(
        (casting_root / "casting_data" / "casting_data" / "train" / "ok_front").glob(
            "*.jpeg"
        )
    )
    casting_test_ok = sorted(
        (casting_root / "casting_data" / "casting_data" / "test" / "ok_front").glob(
            "*.jpeg"
        )
    )
    casting_test_def = sorted(
        (casting_root / "casting_data" / "casting_data" / "test" / "def_front").glob(
            "*.jpeg"
        )
    )
    memory = [
        Sample("casting", "good", 0, path)
        for path in _paths(casting_train, train_limit, seed)
    ]
    negatives = [
        Sample("casting", "good", 0, path)
        for path in _paths(casting_test_ok, test_limit // 2, seed + 1)
    ]
    positives = [
        Sample("casting", "defect", 1, path)
        for path in _paths(casting_test_def, test_limit // 2, seed + 2)
    ]
    query = sorted(positives + negatives, key=lambda item: item.image_path.name)
    scores, labels, _, _, _ = knn_image_scores(extractor, memory, query, 256)
    metrics = binary_metrics(scores, labels)
    rows.append(
        {
            "candidate": "casting",
            "kind": "casting",
            "n_memory": len(memory),
            "n_query": len(query),
            "knn_auroc": metrics["auroc"],
            "knn_aupr": metrics["aupr"],
            "knn_f1": metrics["f1"],
        }
    )
    print(f"{'casting':12s} knn_auroc={metrics['auroc']:.4f}")
    return rows


def _paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    rng = np.random.default_rng(seed)
    index = rng.choice(len(paths), size=limit, replace=False)
    return [paths[i] for i in sorted(index)]
