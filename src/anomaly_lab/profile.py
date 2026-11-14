from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from .config import IMAGE_SIZE, MVTEC_DIR, RESULTS_DIR
from .data import list_mvtec_samples, load_rgb
from .evaluation import load_cae_checkpoint, load_feature_checkpoint
from .features import PatchFeatureExtractor, build_backbone
from .patches import assemble, extract_patches
from .train import time_prediction


def count_flops(module: torch.nn.Module, example: torch.Tensor) -> float:
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        module(example)
    return float(counter.get_total_flops())


def feature_latency(model, config, samples, device: str) -> float:
    extractor = PatchFeatureExtractor(
        backbone=config.backbone,
        patch_size=config.patch_size,
        stride=config.stride,
        upscale=config.upscale,
        image_size=IMAGE_SIZE,
        device=device,
    )
    images = [load_rgb(sample.image_path, IMAGE_SIZE) for sample in samples]

    target = torch.device(device)
    model.to(target)
    model.eval()

    @torch.inference_mode()
    def run() -> None:
        for image in images:
            patches, coords = extract_patches(image, config.patch_size, config.stride)
            features = extractor.extract_patches(patches).to(target)
            scores = model(features)
            assemble(scores.cpu().numpy(), coords, IMAGE_SIZE, config.patch_size)

    run()
    duration = time_prediction(run, repeats=2)
    return duration / len(images) * 1000.0


def cae_latency(model, config, samples, device: str) -> float:
    device_object = torch.device(device)
    model.to(device_object)
    model.eval()
    images = [load_rgb(sample.image_path, IMAGE_SIZE) for sample in samples]

    @torch.inference_mode()
    def run() -> None:
        for image in images:
            patches, coords = extract_patches(image, config.patch_size, config.stride)
            outputs = []
            for start in range(0, len(patches), 128):
                batch = patches[start : start + 128].to(device_object)
                outputs.append(model.pixel_scores(batch).cpu())
            assemble(torch.cat(outputs).numpy(), coords, IMAGE_SIZE, config.patch_size)

    run()
    duration = time_prediction(run, repeats=2)
    return duration / len(images) * 1000.0


def profile_models(category: str = "screw", sample_count: int = 20) -> pd.DataFrame:
    samples = list_mvtec_samples(MVTEC_DIR, category, "test")[:sample_count]
    feature_model, feature_config = load_feature_checkpoint(
        Path("checkpoints/feature_ae_best.pt")
    )
    cae_model, cae_config = load_cae_checkpoint(Path("checkpoints/cae_best.pt"))

    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "cpu"
    rows = []

    input_size = 224 if feature_config.upscale else feature_config.patch_size
    flop_backbone, _ = build_backbone(feature_config.backbone, pretrained=False)
    backbone_flops = count_flops(
        flop_backbone, torch.zeros(1, 3, input_size, input_size)
    )
    ae_flops = count_flops(
        feature_model.autoencoder, torch.zeros(1, feature_model.feature_mean.numel())
    )
    n_patches = len(
        extract_patches(
            torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
            feature_config.patch_size,
            feature_config.stride,
        )[1]
    )
    feature_flops = (backbone_flops + ae_flops) * n_patches / 1e9
    feature_gpu = (
        feature_latency(feature_model, feature_config, samples, "cuda")
        if has_gpu
        else float("nan")
    )
    feature_cpu = feature_latency(feature_model, feature_config, samples, "cpu")
    rows.append(
        {
            "model": "feature_ae",
            "latency_gpu_ms": feature_gpu,
            "latency_cpu_ms": feature_cpu,
            "flops_gflops": feature_flops,
            "image_auroc": float(
                pd.read_csv(RESULTS_DIR / "final_metrics.csv")
                .set_index("model")
                .loc["feature_ae", "image_auroc"]
            ),
            "device": gpu_name,
        }
    )

    cae_patch_flops = count_flops(
        cae_model, torch.zeros(1, 3, cae_config.patch_size, cae_config.patch_size)
    )
    cae_patches = len(
        extract_patches(
            torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
            cae_config.patch_size,
            cae_config.stride,
        )[1]
    )
    cae_flops = cae_patch_flops * cae_patches / 1e9
    cae_gpu = (
        cae_latency(cae_model, cae_config, samples, "cuda") if has_gpu else float("nan")
    )
    cae_model.to("cpu")
    cae_cpu = cae_latency(cae_model, cae_config, samples, "cpu")
    rows.append(
        {
            "model": "cae",
            "latency_gpu_ms": cae_gpu,
            "latency_cpu_ms": cae_cpu,
            "flops_gflops": cae_flops,
            "image_auroc": float(
                pd.read_csv(RESULTS_DIR / "final_metrics.csv")
                .set_index("model")
                .loc["cae", "image_auroc"]
            ),
            "device": gpu_name,
        }
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "profile.csv", index=False)
    return frame
