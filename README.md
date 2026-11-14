# anomaly-lab

Code for laboratory work 4: patch-based unsupervised anomaly detection on MVTec AD.
The pipeline extracts features from frozen pretrained backbones, trains feature-space and
pixel-space autoencoders on normal patches, profiles latency and FLOPs, and benchmarks
PaDiM and PatchCore from anomalib.

## Setup

```bash
uv sync --extra benchmark
```

PyTorch is pinned to the cu126 wheel index in `pyproject.toml`. Without that pin a plain
`uv add` resolves CPU wheels from PyPI.

## Data

Download MVTec AD into `data/mvtec_ad`. The dataset screening additionally expects the
casting product dataset in `data/casting` and PCB-AoI in `data/pcb_aoi`; the rest of the
pipeline needs only MVTec AD. The scripts expect the original folder layout:

```
data/mvtec_ad/screw/train/good/*.png
data/mvtec_ad/screw/test/<defect>/*.png
data/mvtec_ad/screw/ground_truth/<defect>/*_mask.png
```

## Commands

```bash
uv run anomaly-lab screen
uv run anomaly-lab samples --category screw
uv run anomaly-lab train-feature --category screw --epochs 60
uv run anomaly-lab train-cae --category screw --epochs 60
uv run anomaly-lab train-feature-final --category screw
uv run anomaly-lab train-cae-final --category screw --patch-size 128 --stride 64
uv run anomaly-lab evaluate --category screw
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run anomaly-lab benchmark --category screw
uv run anomaly-lab profile --category screw
uv run anomaly-lab figures --category screw
```

The two grid commands write their CSV and histories after every finished configuration and
accept `--resume`, so an interrupted run continues from the last saved configuration.
The final commands fit and save the checkpoints used by `evaluate` with the configuration
documented in `reports/decisions.md`. `train-cae-final` needs the patch and stride flags
because its defaults are the reference configuration, not the selected one.

`benchmark` installs anomalib through the optional group. Its allocator flag avoids
fragmentation on a 6 GB card; the command uses batch 8 for PaDiM and batch 4 for PatchCore.

Results are written to `results/`, figures to `reports/figures/`, and checkpoints to
`checkpoints/`. The reasoning behind the dataset choice, model design, and evaluation
protocol is documented in `reports/decisions.md`.

## Layout

```
src/anomaly_lab/
  data.py         MVTec reader, normal train/validation split
  patches.py      patch extraction, score-map assembly, smoothing
  features.py     pretrained backbones and feature caching
  models.py       feature autoencoder and convolutional autoencoder
  train.py        training loops, prediction, evaluation helpers
  metrics.py      AUROC, AUPR, best F1, pixel metrics
  screening.py    candidate dataset statistics and kNN baseline
  evaluation.py   checkpoint loading, final metrics, calibrated F1
  profile.py      latency and FLOPs measurement
  benchmark.py    anomalib PaDiM and PatchCore runs
  plots.py        report figures
  report.py       figure assembly from saved results
  cli.py          command line entry point
```
