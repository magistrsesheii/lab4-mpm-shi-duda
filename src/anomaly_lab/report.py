from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import FIGURES_DIR, RESULTS_DIR
from .data import list_mvtec_samples


def _key_feature(row: pd.Series) -> str:
    return (
        f"fae_{row['backbone']}_p{int(row['patch_size'])}_s{int(row['stride'])}_u{int(row['upscale'])}"
        f"_h{int(row['hidden_dim'])}_l{int(row['latent_dim'])}_d{row['dropout']}_wd{row['weight_decay']}"
    )


def _key_cae(row: pd.Series) -> str:
    return f"cae_p{int(row['patch_size'])}_s{int(row['stride'])}_lc{int(row['latent_channels'])}_bc{int(row['base_channels'])}_wd{row['weight_decay']}"


def make_figures(category: str, feature_maps_file: str = "feature_maps.npz") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    samples = list_mvtec_samples(Path("data/mvtec_ad"), category, "test")

    maps_data = np.load(RESULTS_DIR / feature_maps_file, allow_pickle=True)
    maps = maps_data["maps"]
    scores = maps_data["scores"]
    plots.plot_score_heatmaps(samples, maps, scores, count=6)
    cae_maps_path = RESULTS_DIR / "cae_maps.npz"
    if cae_maps_path.exists():
        cae_maps_data = np.load(cae_maps_path, allow_pickle=True)
        plots.plot_score_heatmaps(
            samples,
            cae_maps_data["maps"],
            cae_maps_data["scores"],
            count=6,
            name="fig16_cae_score_heatmaps.png",
            label="convolutional autoencoder",
        )

    feature_grid = pd.read_csv(RESULTS_DIR / "feature_ae_grid.csv")
    cae_grid = pd.read_csv(RESULTS_DIR / "cae_grid.csv")
    histories = {}
    for label, final_name in (
        ("feature ae", "feature_ae_final.json"),
        ("conv ae", "cae_final.json"),
    ):
        path = RESULTS_DIR / final_name
        if path.exists():
            with path.open() as handle:
                histories[label] = json.load(handle)["history"]
    if len(histories) < 2:
        with (RESULTS_DIR / "feature_ae_histories.json").open() as handle:
            feature_histories = json.load(handle)
        with (RESULTS_DIR / "cae_histories.json").open() as handle:
            cae_histories = json.load(handle)
        best_feature = feature_grid.sort_values("best_val_loss").iloc[0]
        best_cae = cae_grid.sort_values("best_val_loss").iloc[0]
        histories = {
            "feature ae": feature_histories[_key_feature(best_feature)],
            "conv ae": cae_histories[_key_cae(best_cae)],
        }
    plots.plot_training_curves(histories)
    plots.plot_patch_backbone(feature_grid)
    plots.plot_regularization(feature_grid)
    plots.plot_cae_grid(cae_grid)
    plots.plot_validation_gap(feature_grid)
    plots.plot_per_defect(
        {
            "feature ae": pd.read_csv(RESULTS_DIR / "predictions_feature_ae.csv"),
            "conv ae": pd.read_csv(RESULTS_DIR / "predictions_cae.csv"),
        }
    )

    score_data = np.load(RESULTS_DIR / "scores_feature_ae.npz", allow_pickle=True)
    final_frame = pd.read_csv(RESULTS_DIR / "final_metrics.csv").set_index("model")
    threshold = float(final_frame.loc["feature_ae", "threshold_calibrated"])
    distributions = {
        "feature ae": (score_data["scores"], score_data["labels"], threshold)
    }
    cae_scores_path = RESULTS_DIR / "scores_cae.npz"
    if cae_scores_path.exists():
        cae_data = np.load(cae_scores_path, allow_pickle=True)
        distributions["conv ae"] = (
            cae_data["scores"],
            cae_data["labels"],
            float(final_frame.loc["cae", "threshold_calibrated"]),
        )
    plots.plot_score_distribution(distributions)
    calibration_path = RESULTS_DIR / "calibration_sweep.csv"
    if calibration_path.exists():
        plots.plot_calibration_sweep(pd.read_csv(calibration_path))

    curves = {}
    for name, label in [
        ("feature_ae", "feature ae"),
        ("cae", "conv ae"),
        ("anomalib_padim", "PaDiM"),
        ("anomalib_patchcore", "PatchCore"),
    ]:
        path = RESULTS_DIR / f"scores_{name}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            curves[label] = (data["scores"], data["labels"], data.get("defects"))
    plots.plot_roc_curves(curves)

    final = pd.read_csv(RESULTS_DIR / "final_metrics.csv")
    comparison_rows = []
    for _, row in final.iterrows():
        comparison_rows.append(
            {
                "model": "feature ae" if row["model"] == "feature_ae" else "conv ae",
                "image_auroc": row["image_auroc"],
                "pixel_auroc": row["pixel_auroc"],
                "f1": row["image_f1"],
            }
        )
    benchmark_path = RESULTS_DIR / "anomalib_benchmark.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        for _, row in benchmark.iterrows():
            comparison_rows.append(
                {
                    "model": "PaDiM" if row["model"] == "padim" else "PatchCore",
                    "image_auroc": row["image_auroc"],
                    "pixel_auroc": row["pixel_auroc"],
                    "f1": row.get("f1_best", row["image_f1"]),
                }
            )
    plots.plot_model_comparison(pd.DataFrame(comparison_rows))

    profile_path = RESULTS_DIR / "profile.csv"
    if profile_path.exists():
        plots.plot_latency_flops(pd.read_csv(profile_path))
