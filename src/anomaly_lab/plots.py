from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR
from .data import Sample, load_mask, load_rgb
from .metrics import roc_points

COLORS = ["#3b6ea5", "#c0504d", "#4f8a5b", "#d9a441", "#7a6fa8", "#5aa6a6", "#9c6b4f"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )


def save_figure(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_screening(
    metadata: pd.DataFrame, category_screening: pd.DataFrame
) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))

    counts = (
        metadata[metadata["dataset"] == "mvtec"]
        .groupby("category")[["train_normal", "test_anomaly"]]
        .sum()
    )
    order = counts.sort_values("train_normal", ascending=False).index
    x = np.arange(len(order))
    axes[0, 0].bar(
        x, counts.loc[order, "train_normal"], color=COLORS[0], label="normal train"
    )
    axes[0, 0].bar(
        x,
        counts.loc[order, "test_anomaly"],
        bottom=counts.loc[order, "train_normal"],
        color=COLORS[1],
        label="anomalous test",
    )
    axes[0, 0].set_xticks(x, order, rotation=60, ha="right")
    axes[0, 0].set_ylabel("images")
    axes[0, 0].set_title("a. MVTec AD category sizes")
    axes[0, 0].legend(loc="upper right")

    anomaly_fraction = metadata.set_index("category")["anomaly_fraction"].sort_values()
    axes[0, 1].barh(anomaly_fraction.index, anomaly_fraction.values, color=COLORS[2])
    axes[0, 1].set_xlabel("fraction of anomalous test images")
    axes[0, 1].set_title("b. class imbalance in each candidate")

    ranked = category_screening.sort_values("knn_auroc")
    colors = [
        COLORS[1] if value < 0.8 else COLORS[3] if value < 0.9 else COLORS[2]
        for value in ranked["knn_auroc"]
    ]
    axes[1, 0].barh(ranked["candidate"], ranked["knn_auroc"], color=colors)
    axes[1, 0].axvline(0.8, color="0.5", linestyle="--", linewidth=1)
    axes[1, 0].set_xlim(0.4, 1.0)
    axes[1, 0].set_xlabel("image AUROC of kNN baseline")
    axes[1, 0].set_title("c. unsupervised difficulty on a fixed baseline")

    area = metadata[metadata["dataset"] == "mvtec"].set_index("category")
    axes[1, 1].scatter(
        area["defect_imbalance"],
        area["mean_mask_area"] * 100,
        s=30 + area["std_mask_area"] * 400,
        c=[COLORS[0] if family == "object" else COLORS[3] for family in area["family"]],
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [
        (4, 3),
        (4, -10),
        (-4, 3),
        (-4, -10),
        (4, 12),
        (4, -19),
        (-4, 12),
        (-4, -19),
        (12, 4),
        (-12, 4),
        (12, -8),
        (-12, -8),
    ]
    for category, row in area.iterrows():
        point = (row["defect_imbalance"], row["mean_mask_area"] * 100)
        label = None
        for dx, dy in offsets:
            candidate = axes[1, 1].annotate(
                category,
                point,
                fontsize=6.5,
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
            )
            bbox = candidate.get_window_extent(renderer)
            if not any(bbox.overlaps(other) for other in placed):
                label = candidate
                placed.append(bbox)
                break
            candidate.remove()
        if label is None:
            label = axes[1, 1].annotate(
                category,
                point,
                fontsize=6.5,
                xytext=(4, 3),
                textcoords="offset points",
            )
            placed.append(label.get_window_extent(renderer))
    axes[1, 1].set_xlabel("largest / smallest defect type")
    axes[1, 1].set_ylabel("mean mask area, percent")
    axes[1, 1].set_title("d. defect diversity and mask size")
    fig.tight_layout()
    return save_figure(fig, "fig01_dataset_screening.png")


def plot_category_ranking(screening: pd.DataFrame) -> Path:
    apply_style()
    data = screening.sort_values("knn_auroc")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [COLORS[1] if name == "screw" else COLORS[0] for name in data["candidate"]]
    ax.barh(data["candidate"], data["knn_auroc"], color=colors)
    for index, value in enumerate(data["knn_auroc"]):
        ax.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=7)
    ax.axvline(0.8, color="0.6", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("image AUROC")
    ax.set_title("kNN baseline on normal patch features, selected category in red")
    return save_figure(fig, "fig02_category_ranking.png")


def plot_dataset_samples(
    samples: list[Sample], columns: int = 5, name: str = "fig03_dataset_samples.png"
) -> Path:
    apply_style()
    chosen = []
    seen = set()
    for sample in samples:
        if sample.defect not in seen:
            seen.add(sample.defect)
            chosen.append(sample)
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, sample in zip(axes, chosen):
        image = load_rgb(sample.image_path).permute(1, 2, 0).numpy()
        mask = load_mask(sample.mask_path).squeeze(0).numpy()
        axis.imshow(image)
        if mask.max() > 0:
            overlay = np.zeros((*mask.shape, 4))
            overlay[..., 0] = 1.0
            overlay[..., 3] = mask * 0.45
            axis.imshow(overlay)
        axis.set_title(sample.defect)
    return save_figure(fig, name)


def plot_training_curves(
    histories: dict[str, list[dict[str, float]]],
    name: str = "fig05_training_curves.png",
) -> Path:
    apply_style()
    count = len(histories)
    fig, axes = plt.subplots(1, count, figsize=(4.6 * count, 3.4), squeeze=False)
    axes = axes[0]
    for index, (label, history) in enumerate(histories.items()):
        epochs = [row["epoch"] for row in history]
        train = [row["train_loss"] for row in history]
        val = [row["val_loss"] for row in history]
        axes[index].plot(epochs, train, color=COLORS[0], label="train")
        axes[index].plot(epochs, val, color=COLORS[1], label="validation")
        best_epoch = int(np.argmin(val)) + 1
        axes[index].axvline(best_epoch, color="0.6", linestyle=":", linewidth=1)
        axes[index].set_yscale("log")
        axes[index].set_xlabel("epoch")
        axes[index].set_ylabel("reconstruction loss")
        axes[index].set_title(f"{label}, best epoch {best_epoch}")
        axes[index].legend()
        axes[index].grid(alpha=0.25)
    return save_figure(fig, name)


def plot_patch_backbone(
    frame: pd.DataFrame, name: str = "fig06_patch_backbone.png"
) -> Path:
    apply_style()
    selected = frame[
        (frame["latent_dim"] == 64)
        & (frame["dropout"] == 0.0)
        & (frame["weight_decay"] == 1e-4)
        & (~frame["upscale"])
        & (frame["patch_size"].isin([32, 64, 128]))
    ].drop_duplicates(subset=["backbone", "patch_size"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for index, (backbone, group) in enumerate(selected.groupby("backbone")):
        group = group.sort_values("patch_size")
        axes[0].plot(
            group["patch_size"],
            group["image_auroc"],
            marker="o",
            color=COLORS[index],
            label=backbone,
        )
        axes[1].plot(
            group["patch_size"],
            group["pixel_auroc"],
            marker="o",
            color=COLORS[index],
            label=backbone,
        )
    axes[0].set_ylabel("image AUROC")
    axes[1].set_ylabel("pixel AUROC")
    for axis in axes:
        axis.set_xlabel("patch size, pixels")
        axis.set_xticks([32, 64, 128])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_title("a. image-level detection")
    axes[1].set_title("b. pixel-level localisation")
    return save_figure(fig, name)


def plot_regularization(
    frame: pd.DataFrame, name: str = "fig07_regularization.png"
) -> Path:
    apply_style()
    base = frame[
        (frame["backbone"] == "resnet18")
        & (frame["patch_size"] == 64)
        & (frame["stride"] == 32)
        & (~frame["upscale"])
    ]
    latent = base[
        (base["weight_decay"] == 1e-4) & (base["dropout"] == 0.0)
    ].sort_values("latent_dim")
    decay = (
        base[(base["latent_dim"] == 64) & (base["dropout"] == 0.0)]
        .drop_duplicates("weight_decay")
        .sort_values("weight_decay")
    )
    dropout = (
        base[(base["latent_dim"] == 64) & (base["weight_decay"] == 1e-4)]
        .drop_duplicates("dropout")
        .sort_values("dropout")
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    axes[0].plot(
        latent["latent_dim"].astype(str),
        latent["image_auroc"],
        marker="o",
        color=COLORS[0],
    )
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("image AUROC")
    axes[0].set_title("a. bottleneck width")
    axes[1].plot(
        decay["weight_decay"].astype(str),
        decay["image_auroc"],
        marker="o",
        color=COLORS[1],
    )
    axes[1].set_xlabel("weight decay")
    axes[1].set_title("b. weight decay")
    axes[2].plot(
        dropout["dropout"].astype(str),
        dropout["image_auroc"],
        marker="o",
        color=COLORS[2],
    )
    axes[2].set_xlabel("dropout")
    axes[2].set_title("c. dropout")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0.795, 0.85)
    return save_figure(fig, name)


def plot_cae_grid(frame: pd.DataFrame, name: str = "fig08_cae_grid.png") -> Path:
    apply_style()
    base = frame.copy()
    patch = (
        base[
            (base["latent_channels"] == 32)
            & (base["base_channels"] == 32)
            & (base["weight_decay"] == 1e-4)
        ]
        .drop_duplicates("patch_size")
        .sort_values("patch_size")
    )
    latent = (
        base[
            (base["patch_size"] == 64)
            & (base["base_channels"] == 32)
            & (base["weight_decay"] == 1e-4)
        ]
        .drop_duplicates("latent_channels")
        .sort_values("latent_channels")
    )
    decay = (
        base[
            (base["patch_size"] == 64)
            & (base["latent_channels"] == 32)
            & (base["base_channels"] == 32)
        ]
        .drop_duplicates("weight_decay")
        .sort_values("weight_decay")
    )
    width = (
        base[
            (base["patch_size"] == 64)
            & (base["latent_channels"] == 32)
            & (base["weight_decay"] == 1e-4)
        ]
        .drop_duplicates("base_channels")
        .sort_values("base_channels")
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2))
    axes[0].plot(
        patch["patch_size"].astype(str),
        patch["image_auroc"],
        marker="o",
        color=COLORS[0],
        label="image",
    )
    axes[0].plot(
        patch["patch_size"].astype(str),
        patch["pixel_auroc"],
        marker="o",
        color=COLORS[1],
        label="pixel",
    )
    axes[0].set_xlabel("patch size")
    axes[0].set_title("a. patch size")
    axes[1].plot(
        latent["latent_channels"].astype(str),
        latent["image_auroc"],
        marker="o",
        color=COLORS[0],
        label="image",
    )
    axes[1].plot(
        latent["latent_channels"].astype(str),
        latent["pixel_auroc"],
        marker="o",
        color=COLORS[1],
        label="pixel",
    )
    axes[1].set_xlabel("latent channels")
    axes[1].set_title("b. bottleneck width")
    axes[2].plot(
        decay["weight_decay"].astype(str),
        decay["image_auroc"],
        marker="o",
        color=COLORS[0],
        label="image",
    )
    axes[2].plot(
        decay["weight_decay"].astype(str),
        decay["pixel_auroc"],
        marker="o",
        color=COLORS[1],
        label="pixel",
    )
    axes[2].set_xlabel("weight decay")
    axes[2].set_title("c. weight decay")
    axes[3].plot(
        width["base_channels"].astype(str),
        width["image_auroc"],
        marker="o",
        color=COLORS[0],
        label="image",
    )
    axes[3].plot(
        width["base_channels"].astype(str),
        width["pixel_auroc"],
        marker="o",
        color=COLORS[1],
        label="pixel",
    )
    axes[3].set_xlabel("base channels")
    axes[3].set_title("d. base width")
    for axis in axes:
        axis.set_ylabel("AUROC")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    return save_figure(fig, name)


def plot_score_distribution(
    distributions: dict[str, tuple[np.ndarray, np.ndarray, float]],
    name: str = "fig09_score_distribution.png",
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, len(distributions), figsize=(10.4, 3.5))
    for axis, (label, (scores, labels, threshold)) in zip(
        np.atleast_1d(axes), distributions.items()
    ):
        bins = np.linspace(float(np.min(scores)), float(np.max(scores)), 40)
        axis.hist(
            scores[labels == 0], bins=bins, alpha=0.7, color=COLORS[0], label="normal"
        )
        axis.hist(
            scores[labels == 1],
            bins=bins,
            alpha=0.7,
            color=COLORS[1],
            label="anomalous",
        )
        axis.axvline(
            threshold, color="0.2", linestyle="--", label="calibrated threshold"
        )
        axis.set_xlabel("image anomaly score")
        axis.set_ylabel("count")
        axis.set_title(f"{label}: score separation on the test set")
        axis.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_calibration_sweep(
    frame: pd.DataFrame, name: str = "fig15_calibration.png"
) -> Path:
    apply_style()
    models = frame["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(10.4, 3.5), sharey=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        subset = frame[frame["model"] == model]
        for index, (normal_set, group) in enumerate(subset.groupby("normal_set")):
            group = group.sort_values("quantile")
            label = (
                "all 320 normal images"
                if normal_set == "all normal"
                else "48 validation images"
            )
            axis.plot(
                group["quantile"],
                group["f1"],
                marker="o",
                color=COLORS[index],
                label=label,
            )
        axis.set_xlabel("threshold quantile of normal scores")
        axis.set_ylabel("test F1")
        axis.set_title(model.replace("_", " "))
        axis.set_ylim(-0.02, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    name: str = "fig10_roc.png",
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for index, (label, (scores, labels, _)) in enumerate(curves.items()):
        fpr, tpr = roc_points(scores, labels)
        ax.plot(fpr, tpr, color=COLORS[index % len(COLORS)], label=label)
    ax.plot([0, 1], [0, 1], color="0.7", linestyle=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("image-level ROC")
    ax.legend(fontsize=8)
    return save_figure(fig, name)


def plot_score_heatmaps(
    samples: list[Sample],
    maps: list[np.ndarray],
    scores: np.ndarray,
    count: int = 6,
    name: str = "fig04_score_heatmaps.png",
    label: str = "feature autoencoder",
) -> Path:
    apply_style()
    chosen = []
    seen = set()
    for index, sample in enumerate(samples):
        if sample.label == 1 and sample.defect not in seen:
            seen.add(sample.defect)
            chosen.append(index)
        if len(chosen) >= count:
            break
    fig, axes = plt.subplots(len(chosen), 4, figsize=(9.0, 2.3 * len(chosen)))
    fig.suptitle(f"{label}: anomaly maps on defective test images")
    axes = np.atleast_2d(axes)
    for row, index in enumerate(chosen):
        sample = samples[index]
        image = load_rgb(sample.image_path).permute(1, 2, 0).numpy()
        mask = load_mask(sample.mask_path).squeeze(0).numpy()
        score_map = maps[index]
        upper = float(np.quantile(score_map, 0.995))
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{sample.defect} ({scores[index]:.3g})")
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("ground truth")
        axes[row, 2].imshow(score_map, cmap="magma", vmin=0.0, vmax=upper)
        axes[row, 2].set_title("anomaly map")
        axes[row, 3].imshow(image)
        axes[row, 3].imshow(score_map, cmap="magma", alpha=0.45, vmin=0.0, vmax=upper)
        axes[row, 3].set_title("overlay")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, name)


def plot_model_comparison(
    frame: pd.DataFrame, name: str = "fig11_model_comparison.png"
) -> Path:
    apply_style()
    metrics = ["image_auroc", "pixel_auroc", "f1"]
    labels = frame["model"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for index, metric in enumerate(metrics):
        ax.bar(
            x + (index - 1) * width,
            frame[metric],
            width,
            color=COLORS[index],
            label=metric.replace("_", " "),
        )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("score")
    ax.set_title("our model versus SOTA baselines")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, name)


def plot_latency_flops(
    frame: pd.DataFrame, name: str = "fig12_latency_flops.png"
) -> Path:
    apply_style()
    rows = [
        ("latency_gpu_ms", "ms per image", "a. GPU latency"),
        ("latency_cpu_ms", "ms per image", "b. CPU latency"),
        ("flops_gflops", "GFLOPs per image", "c. computational cost"),
    ]
    labels = [model.replace("_", " ") for model in frame["model"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for axis, (column, unit, title) in zip(axes, rows):
        values = frame[column].to_numpy()
        axis.bar(x, values, color=COLORS[: len(labels)], width=0.55)
        for position, value in zip(x, values):
            axis.text(
                position,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.set_ylim(0, float(np.nanmax(values)) * 1.18)
        axis.grid(axis="y", alpha=0.25)
    return save_figure(fig, name)


def plot_validation_gap(
    frame: pd.DataFrame, name: str = "fig13_validation_gap.png"
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    backbones = list(frame["backbone"].dropna().unique())
    for index, backbone in enumerate(backbones):
        group = frame[frame["backbone"] == backbone]
        ax.scatter(
            group["best_val_loss"],
            group["image_auroc"],
            s=30 + group["patch_size"] / 2,
            color=COLORS[index % len(COLORS)],
            label=backbone,
            alpha=0.85,
        )
    for _, row in frame.iterrows():
        if row["best_val_loss"] < 0.09 or row["image_auroc"] > 0.84:
            ax.annotate(
                f"p{int(row['patch_size'])} l{int(row['latent_dim'])}",
                (row["best_val_loss"], row["image_auroc"]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("best validation reconstruction loss")
    ax.set_ylabel("test image AUROC")
    ax.set_title("validation loss does not rank detection quality")
    ax.legend(title="backbone", fontsize=7)
    ax.grid(alpha=0.25)
    return save_figure(fig, name)


def plot_per_defect(
    predictions: dict[str, pd.DataFrame], name: str = "fig14_per_defect_scores.png"
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, len(predictions), figsize=(10.4, 3.6))
    for axis, (label, frame) in zip(np.atleast_1d(axes), predictions.items()):
        summary = (
            frame.groupby("defect")["score"].agg(["mean", "std"]).sort_values("mean")
        )
        colors = [
            COLORS[2] if defect == "good" else COLORS[1] for defect in summary.index
        ]
        axis.bar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            color=colors,
            capsize=2,
        )
        axis.axhline(
            float(frame.loc[frame["label"] == 0, "score"].mean()),
            color="0.3",
            linestyle="--",
            label="mean normal score",
        )
        axis.set_ylabel("mean image anomaly score")
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, name)
