from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import MVTEC_DIR, RESULTS_DIR
from .metrics import binary_metrics, pixel_metrics
from .patches import top_fraction_mean


def _load_anomalib():
    from anomalib.data import MVTecAD
    from anomalib.data.utils.split import TestSplitMode, ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore
    from anomalib.post_processing import PostProcessor

    return MVTecAD, TestSplitMode, ValSplitMode, Engine, Padim, Patchcore, PostProcessor


def build_datamodule(category: str, batch_size: int):
    MVTecAD, TestSplitMode, ValSplitMode, _, _, _, _ = _load_anomalib()
    return MVTecAD(
        root=MVTEC_DIR,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.15,
        seed=0,
    )


def collect_predictions(
    engine, model, datamodule
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    batches = engine.predict(
        model=model, datamodule=datamodule, return_predictions=True
    )
    scores = []
    labels = []
    maps = []
    masks = []
    for batch in batches or []:
        items = batch if isinstance(batch, list) else [batch]
        for item in items:
            scores.append(item["pred_score"].detach().cpu().numpy().reshape(-1))
            labels.append(item["gt_label"].detach().cpu().numpy().reshape(-1))
            anomaly_map = item["anomaly_map"].detach().cpu().numpy()
            mask = item["gt_mask"]
            if mask is None:
                mask = np.zeros_like(anomaly_map)
            else:
                mask = mask.detach().cpu().numpy()
            for index in range(anomaly_map.shape[0]):
                item_map = (
                    anomaly_map[index, 0]
                    if anomaly_map.ndim == 4
                    else anomaly_map[index]
                )
                item_mask = mask[index, 0] if mask.ndim == 4 else mask[index]
                maps.append(item_map)
                masks.append((item_mask > 0.5).astype(np.uint8))
    return (
        np.concatenate(scores),
        np.concatenate(labels).astype(np.int64),
        maps,
        masks,
    )


def run_anomalib_model(
    name: str, category: str, batch_size: int = 8, device: str = "cuda"
) -> dict[str, object]:
    _, _, _, Engine, Padim, Patchcore, PostProcessor = _load_anomalib()
    post_processor = PostProcessor(enable_normalization=False)
    if name == "padim":
        model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            post_processor=post_processor,
        )
    elif name == "patchcore":
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
            post_processor=post_processor,
        )
    else:
        raise ValueError(f"unknown anomalib model: {name}")

    datamodule = build_datamodule(category, batch_size)
    engine = Engine(
        max_epochs=1,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        default_root_dir=str(RESULTS_DIR / "anomalib_runs"),
    )
    start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - start
    results = engine.test(model=model, datamodule=datamodule, verbose=False)
    result = results[0] if results else {}
    scores, labels, maps, masks = collect_predictions(engine, model, datamodule)
    metrics = {**binary_metrics(scores, labels), **pixel_metrics(maps, masks)}
    top_scores = np.asarray([top_fraction_mean(item) for item in maps])
    top_metrics = binary_metrics(top_scores, labels)
    np.savez(
        RESULTS_DIR / f"scores_anomalib_{name}.npz",
        scores=scores,
        labels=labels,
    )
    row = {
        "model": name,
        "image_auroc": metrics["auroc"],
        "image_f1": metrics["f1"],
        "f1_best": metrics["f1"],
        "image_aupr": metrics["aupr"],
        "pixel_auroc": metrics["pixel_auroc"],
        "pixel_aupr": metrics["pixel_aupr"],
        "pixel_f1": metrics["pixel_f1"],
        "fit_seconds": fit_seconds,
        "n_scores": len(scores),
        "image_auroc_top1": top_metrics["auroc"],
        "image_aupr_top1": top_metrics["aupr"],
        "anomalib_metrics": str(result),
    }
    return row


def run_benchmark(category: str, device: str = "cuda") -> pd.DataFrame:
    rows = []
    for name in ("padim", "patchcore"):
        print(f"running {name}")
        batch_size = 4 if name == "patchcore" else 8
        rows.append(
            run_anomalib_model(name, category, batch_size=batch_size, device=device)
        )
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "anomalib_benchmark.csv", index=False)
        print(rows[-1])
    return pd.DataFrame(rows)
