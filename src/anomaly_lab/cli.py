from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import plots
from .benchmark import run_benchmark
from .config import (
    CASTING_DIR,
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MVTEC_CATEGORIES,
    MVTEC_DIR,
    RESULTS_DIR,
    ensure_dirs,
)
from .data import list_mvtec_samples, split_train_normal
from .evaluation import evaluate_all
from .features import PatchFeatureExtractor
from .profile import profile_models
from .report import make_figures
from .screening import metadata_table, screen_candidates
from .train import (
    CAEConfig,
    FeatureConfig,
    evaluate_cae_model,
    evaluate_feature_model,
    samples_to_patches,
    train_cae,
    train_feature_ae,
)


def command_screen(args: argparse.Namespace) -> None:
    ensure_dirs()
    metadata = pd.DataFrame(metadata_table())
    metadata.to_csv(RESULTS_DIR / "dataset_screening.csv", index=False)
    categories = args.categories or MVTEC_CATEGORIES
    screening = pd.DataFrame(
        screen_candidates(
            MVTEC_DIR,
            args.casting_root,
            categories,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            device=args.device,
        )
    )
    screening.to_csv(RESULTS_DIR / "category_screening.csv", index=False)
    plots.plot_dataset_screening(metadata, screening)
    plots.plot_category_ranking(screening)
    print(screening.sort_values("knn_auroc").to_string(index=False))


def command_samples(args: argparse.Namespace) -> None:
    ensure_dirs()
    samples = list_mvtec_samples(MVTEC_DIR, args.category, "test")
    good = [s for s in samples if s.label == 0][:1]
    defects = []
    seen = set()
    for sample in samples:
        if sample.label == 1 and sample.defect not in seen:
            seen.add(sample.defect)
            defects.append(sample)
    chosen = good + defects
    if args.limit:
        chosen = chosen[: args.limit]
    plots.plot_dataset_samples(
        chosen, columns=min(3, len(chosen)), name="fig03_dataset_samples.png"
    )
    print(f"wrote sample grid for {args.category}")


def command_train_feature(args: argparse.Namespace) -> None:
    ensure_dirs()
    category = args.category
    train_samples = list_mvtec_samples(MVTEC_DIR, category, "train")
    test_samples = list_mvtec_samples(MVTEC_DIR, category, "test")
    train_split, val_split = split_train_normal(
        train_samples, args.val_fraction, seed=0
    )
    print(
        f"{category}: train={len(train_split)} val={len(val_split)} test={len(test_samples)}"
    )

    configs = build_feature_grid(args)
    cache_dir = args.cache_dir
    frame_path = RESULTS_DIR / "feature_ae_grid.csv"
    rows, histories = load_grid_state(
        frame_path, RESULTS_DIR / "feature_ae_histories.json", args.resume
    )
    done = {row["name"] for row in rows}
    for config in configs:
        if config.name in done:
            print(f"skip {config.name}")
            continue
        extractor = PatchFeatureExtractor(
            backbone=config.backbone,
            patch_size=config.patch_size,
            stride=config.stride,
            upscale=config.upscale,
            image_size=IMAGE_SIZE,
            device=args.device,
        )
        train_cache = extractor.extract_to_cache(
            cache_dir, category, "train", train_split
        )
        val_cache = extractor.extract_to_cache(cache_dir, category, "val", val_split)
        test_cache = extractor.extract_to_cache(
            cache_dir, category, "test", test_samples
        )

        model, history = train_feature_ae(
            train_cache, val_cache, config, device=args.device
        )
        metrics = evaluate_feature_model(
            model,
            test_cache,
            test_samples,
            config.patch_size,
            IMAGE_SIZE,
            device=args.device,
        )
        best_epoch = int(np.argmin([row["val_loss"] for row in history]) + 1)
        row = {
            "name": config.name,
            "model": "feature_ae",
            "backbone": config.backbone,
            "patch_size": config.patch_size,
            "stride": config.stride,
            "upscale": config.upscale,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "dropout": config.dropout,
            "weight_decay": config.weight_decay,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": float(min(item["val_loss"] for item in history)),
            "image_auroc": metrics["auroc"],
            "image_aupr": metrics["aupr"],
            "image_f1": metrics["f1"],
            "image_threshold": metrics["threshold"],
            "pixel_auroc": metrics["pixel_auroc"],
            "pixel_f1": metrics["pixel_f1"],
        }
        rows.append(row)
        histories[config.name] = history
        pd.DataFrame(rows).sort_values("best_val_loss").to_csv(frame_path, index=False)
        save_histories(histories, "feature_ae_histories.json")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": row,
                "category": category,
            },
            CHECKPOINTS_DIR / f"feature_ae_{config.name}.pt",
        )
        print(
            f"{config.name} val={row['best_val_loss']:.4f} image_auroc={row['image_auroc']:.4f} "
            f"pixel_auroc={row['pixel_auroc']:.4f}"
        )

    frame = pd.DataFrame(rows).sort_values("best_val_loss")
    frame.to_csv(frame_path, index=False)
    best_config = frame.iloc[0]
    best_key = select_history_key(histories, best_config)
    best_checkpoint = torch.load(
        CHECKPOINTS_DIR / f"feature_ae_{best_key}.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(
        {
            "state_dict": best_checkpoint["state_dict"],
            "config": best_config.to_dict(),
            "category": category,
        },
        CHECKPOINTS_DIR / "feature_ae_best.pt",
    )
    print(frame.to_string(index=False))


def command_train_feature_final(args: argparse.Namespace) -> None:
    ensure_dirs()
    category = args.category
    config = FeatureConfig(
        backbone=args.backbone,
        patch_size=args.patch_size,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )
    train_samples = list_mvtec_samples(MVTEC_DIR, category, "train")
    test_samples = list_mvtec_samples(MVTEC_DIR, category, "test")
    train_split, val_split = split_train_normal(
        train_samples, args.val_fraction, seed=0
    )
    extractor = PatchFeatureExtractor(
        backbone=config.backbone,
        patch_size=config.patch_size,
        stride=config.stride,
        upscale=config.upscale,
        image_size=IMAGE_SIZE,
        device=args.device,
    )
    train_cache = extractor.extract_to_cache(
        args.cache_dir, category, "train", train_split
    )
    val_cache = extractor.extract_to_cache(args.cache_dir, category, "val", val_split)
    test_cache = extractor.extract_to_cache(
        args.cache_dir, category, "test", test_samples
    )
    model, history = train_feature_ae(
        train_cache, val_cache, config, device=args.device
    )
    metrics = evaluate_feature_model(
        model,
        test_cache,
        test_samples,
        config.patch_size,
        IMAGE_SIZE,
        device=args.device,
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": config.backbone,
            "patch_size": config.patch_size,
            "stride": config.stride,
            "upscale": config.upscale,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "dropout": config.dropout,
            "weight_decay": config.weight_decay,
            "input_dim": model.feature_mean.numel(),
            "category": category,
        },
        "category": category,
    }
    torch.save(checkpoint, CHECKPOINTS_DIR / "feature_ae_best.pt")
    with (RESULTS_DIR / "feature_ae_final.json").open("w") as handle:
        json.dump(
            {"config": checkpoint["config"], "history": history, "metrics": metrics},
            handle,
            indent=1,
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
            indent=1,
        )
    )


def command_train_cae_final(args: argparse.Namespace) -> None:
    ensure_dirs()
    category = args.category
    config = CAEConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )
    train_samples = list_mvtec_samples(MVTEC_DIR, category, "train")
    test_samples = list_mvtec_samples(MVTEC_DIR, category, "test")
    train_split, val_split = split_train_normal(
        train_samples, args.val_fraction, seed=0
    )
    train_cache = samples_to_patches(
        train_split, config.patch_size, config.stride, IMAGE_SIZE
    )
    val_cache = samples_to_patches(
        val_split, config.patch_size, config.stride, IMAGE_SIZE
    )
    model, history = train_cae(train_cache, val_cache, config, device=args.device)
    metrics = evaluate_cae_model(
        model,
        test_samples,
        config.patch_size,
        config.stride,
        IMAGE_SIZE,
        device=args.device,
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "patch_size": config.patch_size,
            "stride": config.stride,
            "latent_channels": config.latent_channels,
            "base_channels": config.base_channels,
            "weight_decay": config.weight_decay,
            "category": category,
        },
        "category": category,
    }
    torch.save(checkpoint, CHECKPOINTS_DIR / "cae_best.pt")
    with (RESULTS_DIR / "cae_final.json").open("w") as handle:
        json.dump(
            {"config": checkpoint["config"], "history": history, "metrics": metrics},
            handle,
            indent=1,
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
            indent=1,
        )
    )


def command_train_cae(args: argparse.Namespace) -> None:
    ensure_dirs()
    category = args.category
    train_samples = list_mvtec_samples(MVTEC_DIR, category, "train")
    test_samples = list_mvtec_samples(MVTEC_DIR, category, "test")
    train_split, val_split = split_train_normal(
        train_samples, args.val_fraction, seed=0
    )
    frame_path = RESULTS_DIR / "cae_grid.csv"
    rows, histories = load_grid_state(
        frame_path, RESULTS_DIR / "cae_histories.json", args.resume
    )
    done = {row["name"] for row in rows}
    for config in build_cae_grid(args):
        if config.name in done:
            print(f"skip {config.name}")
            continue
        train_cache = samples_to_patches(
            train_split, config.patch_size, config.stride, IMAGE_SIZE
        )
        val_cache = samples_to_patches(
            val_split, config.patch_size, config.stride, IMAGE_SIZE
        )
        model, history = train_cae(train_cache, val_cache, config, device=args.device)
        metrics = evaluate_cae_model(
            model,
            test_samples,
            config.patch_size,
            config.stride,
            IMAGE_SIZE,
            device=args.device,
        )
        row = {
            "name": config.name,
            "model": "cae",
            "patch_size": config.patch_size,
            "stride": config.stride,
            "latent_channels": config.latent_channels,
            "base_channels": config.base_channels,
            "weight_decay": config.weight_decay,
            "epochs_run": len(history),
            "best_epoch": int(np.argmin([item["val_loss"] for item in history]) + 1),
            "best_val_loss": float(min(item["val_loss"] for item in history)),
            "image_auroc": metrics["auroc"],
            "image_aupr": metrics["aupr"],
            "image_f1": metrics["f1"],
            "image_threshold": metrics["threshold"],
            "pixel_auroc": metrics["pixel_auroc"],
            "pixel_f1": metrics["pixel_f1"],
        }
        rows.append(row)
        histories[config.name] = history
        pd.DataFrame(rows).sort_values("best_val_loss").to_csv(frame_path, index=False)
        save_histories(histories, "cae_histories.json")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": row,
                "category": category,
            },
            CHECKPOINTS_DIR / f"cae_{config.name}.pt",
        )
        print(
            f"{config.name} val={row['best_val_loss']:.5f} image_auroc={row['image_auroc']:.4f} "
            f"pixel_auroc={row['pixel_auroc']:.4f}"
        )

    frame = pd.DataFrame(rows).sort_values("best_val_loss")
    frame.to_csv(frame_path, index=False)
    best_key = select_history_key(histories, frame.iloc[0])
    best_checkpoint = torch.load(
        CHECKPOINTS_DIR / f"cae_{best_key}.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(
        {
            "state_dict": best_checkpoint["state_dict"],
            "config": frame.iloc[0].to_dict(),
            "category": category,
        },
        CHECKPOINTS_DIR / "cae_best.pt",
    )
    print(frame.to_string(index=False))


def select_history_key(
    histories: dict[str, list[dict[str, float]]], row: pd.Series
) -> str:
    if row["model"] == "cae":
        return f"cae_p{int(row['patch_size'])}_s{int(row['stride'])}_lc{int(row['latent_channels'])}_bc{int(row['base_channels'])}_wd{row['weight_decay']}"
    return (
        f"fae_{row['backbone']}_p{int(row['patch_size'])}_s{int(row['stride'])}_u{int(row['upscale'])}"
        f"_h{int(row['hidden_dim'])}_l{int(row['latent_dim'])}_d{row['dropout']}_wd{row['weight_decay']}"
    )


def save_histories(histories: dict[str, list[dict[str, float]]], name: str) -> None:
    with (RESULTS_DIR / name).open("w") as handle:
        json.dump(histories, handle, indent=1)


def load_grid_state(
    frame_path: Path, histories_path: Path, resume: bool
) -> tuple[list[dict], dict[str, list[dict[str, float]]]]:
    if not resume or not frame_path.exists():
        return [], {}
    rows = pd.read_csv(frame_path).to_dict("records")
    if not rows or "name" not in rows[0]:
        return [], {}
    histories = {}
    if histories_path.exists():
        with histories_path.open() as handle:
            histories = json.load(handle)
    return rows, histories


def build_feature_grid(args: argparse.Namespace) -> list[FeatureConfig]:
    defaults = {"epochs": args.epochs, "patience": args.patience, "lr": args.lr}
    configs = []
    for backbone in ("resnet18", "mobilenet_v3_small"):
        for patch_size in (32, 64, 128):
            configs.append(
                FeatureConfig(
                    backbone=backbone,
                    patch_size=patch_size,
                    stride=patch_size // 2,
                    hidden_dim=256,
                    latent_dim=64,
                    weight_decay=1e-4,
                    **defaults,
                )
            )
    configs.append(FeatureConfig(patch_size=32, stride=16, upscale=True, **defaults))
    for latent_dim in (16, 32, 128, 256):
        configs.append(
            FeatureConfig(patch_size=64, stride=32, latent_dim=latent_dim, **defaults)
        )
    for weight_decay in (0.0, 1e-3, 1e-2):
        configs.append(
            FeatureConfig(
                patch_size=64, stride=32, weight_decay=weight_decay, **defaults
            )
        )
    for dropout in (0.1, 0.3):
        configs.append(
            FeatureConfig(patch_size=64, stride=32, dropout=dropout, **defaults)
        )
    unique = {}
    for config in configs:
        unique[config.name] = config
    if args.quick:
        keep = [
            config
            for config in unique.values()
            if config.patch_size == 64 and config.backbone == "resnet18"
        ]
        return keep
    return list(unique.values())


def build_cae_grid(args: argparse.Namespace) -> list[CAEConfig]:
    defaults = {"epochs": args.epochs, "patience": args.patience, "lr": args.lr}
    configs = []
    for patch_size in (32, 64, 128):
        configs.append(
            CAEConfig(patch_size=patch_size, stride=patch_size // 2, **defaults)
        )
    for latent_channels in (16, 64, 128):
        configs.append(
            CAEConfig(
                patch_size=64, stride=32, latent_channels=latent_channels, **defaults
            )
        )
    for weight_decay in (0.0, 1e-3, 1e-2):
        configs.append(
            CAEConfig(patch_size=64, stride=32, weight_decay=weight_decay, **defaults)
        )
    for base_channels in (16, 64):
        configs.append(
            CAEConfig(patch_size=64, stride=32, base_channels=base_channels, **defaults)
        )
    unique = {}
    for config in configs:
        unique[config.name] = config
    if args.quick:
        keep = [config for config in unique.values() if config.patch_size == 64]
        return keep
    return list(unique.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen", help="compare candidate datasets")
    screen.add_argument("--categories", nargs="*", default=None)
    screen.add_argument("--train-limit", type=int, default=40)
    screen.add_argument("--test-limit", type=int, default=80)
    screen.add_argument("--device", default="cuda")
    screen.add_argument("--casting-root", type=Path, default=CASTING_DIR)
    screen.set_defaults(func=command_screen)

    samples = subparsers.add_parser(
        "samples", help="draw a sample grid for one category"
    )
    samples.add_argument("--category", default="screw")
    samples.add_argument("--limit", type=int, default=6)
    samples.set_defaults(func=command_samples)

    feature = subparsers.add_parser(
        "train-feature", help="train patch feature autoencoders"
    )
    feature.add_argument("--category", default="screw")
    feature.add_argument("--epochs", type=int, default=60)
    feature.add_argument("--patience", type=int, default=12)
    feature.add_argument("--lr", type=float, default=1e-3)
    feature.add_argument("--val-fraction", type=float, default=0.15)
    feature.add_argument("--cache-dir", type=Path, default=Path("data/features"))
    feature.add_argument("--device", default="cuda")
    feature.add_argument("--quick", action="store_true")
    feature.add_argument("--resume", action="store_true")
    feature.set_defaults(func=command_train_feature)

    cae = subparsers.add_parser(
        "train-cae", help="train convolutional patch autoencoders"
    )
    cae.add_argument("--category", default="screw")
    cae.add_argument("--epochs", type=int, default=60)
    cae.add_argument("--patience", type=int, default=12)
    cae.add_argument("--lr", type=float, default=1e-3)
    cae.add_argument("--val-fraction", type=float, default=0.15)
    cae.add_argument("--device", default="cuda")
    cae.add_argument("--quick", action="store_true")
    cae.add_argument("--resume", action="store_true")
    cae.set_defaults(func=command_train_cae)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate trained models and save metrics"
    )
    evaluate.add_argument("--category", default="screw")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--cache-dir", type=Path, default=Path("data/features"))
    evaluate.set_defaults(
        func=lambda args: evaluate_all(args.category, args.device, args.cache_dir)
    )

    profile = subparsers.add_parser("profile", help="measure latency and FLOPs")
    profile.add_argument("--category", default="screw")
    profile.add_argument("--samples", type=int, default=20)
    profile.set_defaults(
        func=lambda args: print(
            profile_models(args.category, args.samples).to_string(index=False)
        )
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="run anomalib PaDiM and PatchCore"
    )
    benchmark.add_argument("--category", default="screw")
    benchmark.add_argument("--device", default="cuda")
    benchmark.set_defaults(
        func=lambda args: print(
            run_benchmark(args.category, args.device).to_string(index=False)
        )
    )

    feature_final = subparsers.add_parser(
        "train-feature-final", help="fit and save the selected feature autoencoder"
    )
    feature_final.add_argument("--category", default="screw")
    feature_final.add_argument("--backbone", default="resnet18")
    feature_final.add_argument("--patch-size", type=int, default=64)
    feature_final.add_argument("--stride", type=int, default=32)
    feature_final.add_argument("--hidden-dim", type=int, default=256)
    feature_final.add_argument("--latent-dim", type=int, default=64)
    feature_final.add_argument("--dropout", type=float, default=0.0)
    feature_final.add_argument("--weight-decay", type=float, default=1e-4)
    feature_final.add_argument("--epochs", type=int, default=60)
    feature_final.add_argument("--patience", type=int, default=12)
    feature_final.add_argument("--lr", type=float, default=1e-3)
    feature_final.add_argument("--val-fraction", type=float, default=0.15)
    feature_final.add_argument("--cache-dir", type=Path, default=Path("data/features"))
    feature_final.add_argument("--device", default="cuda")
    feature_final.set_defaults(func=command_train_feature_final)

    cae_final = subparsers.add_parser(
        "train-cae-final", help="fit and save the selected convolutional autoencoder"
    )
    cae_final.add_argument("--category", default="screw")
    cae_final.add_argument("--patch-size", type=int, default=64)
    cae_final.add_argument("--stride", type=int, default=32)
    cae_final.add_argument("--latent-channels", type=int, default=32)
    cae_final.add_argument("--base-channels", type=int, default=32)
    cae_final.add_argument("--weight-decay", type=float, default=1e-4)
    cae_final.add_argument("--epochs", type=int, default=60)
    cae_final.add_argument("--patience", type=int, default=12)
    cae_final.add_argument("--lr", type=float, default=1e-3)
    cae_final.add_argument("--val-fraction", type=float, default=0.15)
    cae_final.add_argument("--device", default="cuda")
    cae_final.set_defaults(func=command_train_cae_final)

    figures = subparsers.add_parser(
        "figures", help="build report figures from saved results"
    )
    figures.add_argument("--category", default="screw")
    figures.set_defaults(func=lambda args: make_figures(args.category))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
