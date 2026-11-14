from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import IMAGE_SIZE, MVTEC_DIR, RESULTS_DIR
from .data import list_mvtec_samples, split_train_normal
from .features import PatchFeatureExtractor
from .metrics import binary_metrics, pixel_metrics
from .models import ConvAutoEncoder, FeatureAnomalyModel, FeatureAutoEncoder
from .train import (
    CAEConfig,
    FeatureConfig,
    predict_cae_maps,
    predict_feature_maps,
)


def load_feature_checkpoint(path: Path) -> tuple[FeatureAnomalyModel, FeatureConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint["config"]
    config = FeatureConfig(
        backbone=str(raw["backbone"]),
        patch_size=int(raw["patch_size"]),
        stride=int(raw["stride"]),
        upscale=bool(raw["upscale"]),
        hidden_dim=int(raw["hidden_dim"]),
        latent_dim=int(raw["latent_dim"]),
        dropout=float(raw["dropout"]),
        weight_decay=float(raw["weight_decay"]),
    )
    input_dim = int(checkpoint["state_dict"]["feature_mean"].numel())
    autoencoder = FeatureAutoEncoder(
        input_dim, config.hidden_dim, config.latent_dim, config.dropout
    )
    model = FeatureAnomalyModel(
        autoencoder, torch.zeros(input_dim), torch.ones(input_dim)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def load_cae_checkpoint(path: Path) -> tuple[ConvAutoEncoder, CAEConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint["config"]
    config = CAEConfig(
        patch_size=int(raw["patch_size"]),
        stride=int(raw["stride"]),
        latent_channels=int(raw["latent_channels"]),
        base_channels=int(raw["base_channels"]),
        weight_decay=float(raw["weight_decay"]),
    )
    model = ConvAutoEncoder(config.latent_channels, config.base_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def feature_test_cache(
    category: str, config: FeatureConfig, test_samples, cache_dir: Path, device: str
):
    extractor = PatchFeatureExtractor(
        backbone=config.backbone,
        patch_size=config.patch_size,
        stride=config.stride,
        upscale=config.upscale,
        image_size=IMAGE_SIZE,
        device=device,
    )
    return extractor.extract_to_cache(cache_dir, category, "test", test_samples)


def normal_scores_feature(
    model: FeatureAnomalyModel,
    category: str,
    config: FeatureConfig,
    samples,
    split: str,
    cache_dir: Path,
    device: str,
) -> np.ndarray:
    extractor = PatchFeatureExtractor(
        backbone=config.backbone,
        patch_size=config.patch_size,
        stride=config.stride,
        upscale=config.upscale,
        image_size=IMAGE_SIZE,
        device=device,
    )
    cache = extractor.extract_to_cache(cache_dir, category, split, samples)
    scores, labels, _, _, _ = predict_feature_maps(
        model, cache, samples, config.patch_size, IMAGE_SIZE, device
    )
    return scores[labels == 0]


def normal_scores_cae(
    model: ConvAutoEncoder,
    config: CAEConfig,
    samples,
    device: str,
) -> np.ndarray:
    scores, labels, _, _, _ = predict_cae_maps(
        model, samples, config.patch_size, config.stride, IMAGE_SIZE, device
    )
    return scores[labels == 0]


def per_defect_metrics(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    defects: np.ndarray,
    threshold: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    normal = labels == 0
    for defect in sorted(set(defects[labels == 1])):
        detected = defects == defect
        selected = normal | detected
        rows.append(
            {
                "model": name,
                "defect": str(defect),
                "n": int(detected.sum()),
                "auroc_against_good": float(
                    roc_auc_score(labels[selected], scores[selected])
                ),
                "mean_score": float(scores[detected].mean()),
                "detection_rate": float((scores[detected] >= threshold).mean()),
            }
        )
    return rows


def aggregation_sweep(
    maps: list[np.ndarray],
    labels: np.ndarray,
    fractions: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
) -> list[dict[str, float | str]]:
    flat = np.stack([np.asarray(item, dtype=np.float32).ravel() for item in maps])
    rows: list[dict[str, float | str]] = []
    variants: list[tuple[str, np.ndarray]] = []
    for fraction in fractions:
        count = max(1, round(flat.shape[1] * fraction))
        part = np.partition(flat, -count, axis=1)[:, -count:]
        variants.append((f"top {fraction * 100:g} percent", part.mean(axis=1)))
    variants.append(("maximum", flat.max(axis=1)))
    variants.append(("mean", flat.mean(axis=1)))
    for name, scores in variants:
        metrics = binary_metrics(scores, labels)
        rows.append(
            {
                "aggregation": name,
                "image_auroc": metrics["auroc"],
                "image_f1": metrics["f1"],
            }
        )
    return rows


def threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive / max(1, int((labels == 0).sum()))),
    }


def calibrated_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    normal_scores: np.ndarray,
    quantile: float = 0.99,
) -> tuple[float, float]:
    threshold = float(np.quantile(normal_scores, quantile))
    return threshold_metrics(scores, labels, threshold)["f1"], threshold


def calibration_sweep(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    normal_sets: dict[str, np.ndarray],
    quantiles: tuple[float, ...] = (0.8, 0.85, 0.9, 0.95, 0.975, 0.99),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for normal_name, normal_scores in normal_sets.items():
        for quantile in quantiles:
            threshold = float(np.quantile(normal_scores, quantile))
            rows.append(
                {
                    "model": name,
                    "normal_set": normal_name,
                    "quantile": quantile,
                    "threshold": threshold,
                    **threshold_metrics(scores, labels, threshold),
                }
            )
    return rows


def save_scores(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    defects: np.ndarray,
    paths: list[str],
) -> None:
    np.savez(
        RESULTS_DIR / f"scores_{name}.npz",
        scores=scores,
        labels=labels,
        defects=defects.astype(str),
        paths=np.asarray(paths),
    )


def evaluate_all(
    category: str, device: str = "cuda", cache_dir: Path = Path("data/features")
) -> pd.DataFrame:
    train_samples = list_mvtec_samples(MVTEC_DIR, category, "train")
    test_samples = list_mvtec_samples(MVTEC_DIR, category, "test")
    train_split, val_samples = split_train_normal(train_samples, 0.15, seed=0)

    calibration_rows: list[dict[str, float | str]] = []
    rows = []
    feature_model, feature_config = load_feature_checkpoint(
        Path("checkpoints/feature_ae_best.pt")
    )
    feature_model.to(device)
    cache = feature_test_cache(
        category, feature_config, test_samples, cache_dir, device
    )
    scores, labels, maps, masks, defects = predict_feature_maps(
        feature_model,
        cache,
        test_samples,
        feature_config.patch_size,
        IMAGE_SIZE,
        device,
    )
    metrics = {**binary_metrics(scores, labels), **pixel_metrics(maps, masks)}
    train_normal = normal_scores_feature(
        feature_model, category, feature_config, train_split, "train", cache_dir, device
    )
    val_normal = normal_scores_feature(
        feature_model, category, feature_config, val_samples, "val", cache_dir, device
    )
    normal_scores = val_normal
    calibration_rows += calibration_sweep(
        "feature_ae",
        scores,
        labels,
        {
            "validation": val_normal,
            "all normal": np.concatenate([train_normal, val_normal]),
        },
    )
    calibrated, threshold = calibrated_f1(scores, labels, normal_scores)
    defect_rows = per_defect_metrics(
        "feature_ae", scores, labels, defects, metrics["threshold"]
    )
    paths = [sample.image_path for sample in test_samples]
    save_scores("feature_ae", scores, labels, defects, [str(path) for path in paths])
    rows.append(
        {
            "model": "feature_ae",
            "image_auroc": metrics["auroc"],
            "image_aupr": metrics["aupr"],
            "image_f1": metrics["f1"],
            "threshold_best": metrics["threshold"],
            "f1_calibrated": calibrated,
            "threshold_calibrated": threshold,
            "pixel_auroc": metrics["pixel_auroc"],
            "pixel_aupr": metrics["pixel_aupr"],
            "pixel_f1": metrics["pixel_f1"],
        }
    )
    pd.DataFrame(
        {
            "path": paths,
            "defect": defects,
            "label": labels,
            "score": scores,
        }
    ).to_csv(RESULTS_DIR / "predictions_feature_ae.csv", index=False)
    np.savez(
        RESULTS_DIR / "feature_maps.npz",
        maps=np.stack(maps),
        scores=scores,
        labels=labels,
        defects=defects.astype(str),
        paths=np.asarray([str(path) for path in paths]),
    )

    cae_model, cae_config = load_cae_checkpoint(Path("checkpoints/cae_best.pt"))
    cae_model.to(device)
    cae_scores, cae_labels, cae_maps, cae_masks, cae_defects = predict_cae_maps(
        cae_model,
        test_samples,
        cae_config.patch_size,
        cae_config.stride,
        IMAGE_SIZE,
        device,
    )
    cae_metrics = {
        **binary_metrics(cae_scores, cae_labels),
        **pixel_metrics(cae_maps, cae_masks),
    }
    cae_train_normal = normal_scores_cae(cae_model, cae_config, train_split, device)
    cae_val_normal = normal_scores_cae(cae_model, cae_config, val_samples, device)
    cae_normal = cae_val_normal
    calibration_rows += calibration_sweep(
        "cae",
        cae_scores,
        cae_labels,
        {
            "validation": cae_val_normal,
            "all normal": np.concatenate([cae_train_normal, cae_val_normal]),
        },
    )
    cae_calibrated, cae_threshold = calibrated_f1(cae_scores, cae_labels, cae_normal)
    defect_rows += per_defect_metrics(
        "cae", cae_scores, cae_labels, cae_defects, cae_metrics["threshold"]
    )
    save_scores(
        "cae",
        cae_scores,
        cae_labels,
        cae_defects,
        [str(sample.image_path) for sample in test_samples],
    )
    np.savez(
        RESULTS_DIR / "cae_maps.npz",
        maps=np.stack(cae_maps),
        scores=cae_scores,
        labels=cae_labels,
        defects=cae_defects.astype(str),
        paths=np.asarray([str(s.image_path) for s in test_samples]),
    )
    rows.append(
        {
            "model": "cae",
            "image_auroc": cae_metrics["auroc"],
            "image_aupr": cae_metrics["aupr"],
            "image_f1": cae_metrics["f1"],
            "threshold_best": cae_metrics["threshold"],
            "f1_calibrated": cae_calibrated,
            "threshold_calibrated": cae_threshold,
            "pixel_auroc": cae_metrics["pixel_auroc"],
            "pixel_aupr": cae_metrics["pixel_aupr"],
            "pixel_f1": cae_metrics["pixel_f1"],
        }
    )
    pd.DataFrame(
        {
            "path": [sample.image_path for sample in test_samples],
            "defect": cae_defects,
            "label": cae_labels,
            "score": cae_scores,
        }
    ).to_csv(RESULTS_DIR / "predictions_cae.csv", index=False)

    pd.DataFrame(defect_rows).to_csv(
        RESULTS_DIR / "per_defect_metrics.csv", index=False
    )
    aggregation_rows = [
        {"model": "feature_ae", **row} for row in aggregation_sweep(maps, labels)
    ]
    aggregation_rows += [
        {"model": "cae", **row} for row in aggregation_sweep(cae_maps, cae_labels)
    ]
    pd.DataFrame(aggregation_rows).to_csv(
        RESULTS_DIR / "aggregation_ablation.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        RESULTS_DIR / "calibration_sweep.csv", index=False
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "final_metrics.csv", index=False)
    return frame
