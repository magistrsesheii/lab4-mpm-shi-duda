# Laboratory work 4: decisions and results

This file records the choices made while building the anomaly detection pipeline, the
experiments that were run, and the numbers that can be quoted in the report. It is meant
to be a source document for another writer, not a finished report.

## 1. Environment

- Python 3.12, managed with uv (`uv sync --extra benchmark`).
- PyTorch 2.14.1 with CUDA 12.6 wheels, torchvision 0.29.1.
- GPU: Nvidia GTX 1060 6 GB (Pascal, compute capability 6.1). CPU: 12 threads, 15 GB RAM.
- anomalib 2.7.0 is installed as an optional dependency group and used only for the
  PaDiM and PatchCore baselines.

The cu126 index is pinned in `pyproject.toml` under `[tool.uv.sources]`. Without the pin,
adding an unrelated package re-resolves torch from PyPI and silently replaces the CUDA
build with a CPU wheel. This happened once during setup and is the reason the pin exists.

## 2. Dataset selection

Three public sources were considered. The screening numbers come from
`results/dataset_screening.csv` and `results/category_screening.csv`.

### 2.1 Candidates

- MVTec AD (Kaggle mirror `ipythonx/mvtec-ad`). Fifteen industrial categories, pixel-level
  ground truth masks, 320 normal training images for the chosen category, and a separate
  test set with several defect types.
- Casting product image data for quality inspection (Kaggle
  `ravirajsinh45/real-life-industrial-dataset-of-casting-product`). 7348 images at
  300x300, binary labels (ok/defect), no pixel masks. Its training folder is not normal
  only: it holds 2875 ok and 3758 defective images, so fitting the unsupervised model
  means discarding more than half of the training data. The 512x512 upscaled copy was
  ignored because it contains no new information.
- PCB-AoI (Kaggle `kubeedgeianvs/pcb-aoi`). 173 training images and 60 test images, every
  one of them annotated with defect boxes. There is no normal-only training split, so an
  unsupervised model cannot be fitted in the usual way, and the annotations are boxes
  rather than pixel masks.

### 2.2 Screening protocol

Every candidate that supports unsupervised training was scored with the same fixed
baseline: ResNet-18 ImageNet features on 64x64 patches with stride 32, a memory bank of
40 normal training images, and nearest-neighbour cosine distance. The image score is the
mean of the top 1 percent of the assembled patch score map. No model was trained for this
step; it measures how separable normal and anomalous patches already are under a generic
backbone.

Table 1 in the report can use the following values. The full list is in
`results/category_screening.csv`.

| candidate | image AUROC of kNN baseline |
| --- | --- |
| tile | 0.998 |
| bottle | 0.991 |
| leather | 0.987 |
| toothbrush | 0.958 |
| metal_nut | 0.958 |
| hazelnut | 0.956 |
| wood | 0.946 |
| cable | 0.938 |
| zipper | 0.929 |
| transistor | 0.875 |
| pill | 0.830 |
| carpet | 0.829 |
| capsule | 0.808 |
| casting | 0.677 |
| grid | 0.661 |
| screw | 0.602 |

### 2.3 Decision

MVTec AD was selected as the dataset and the screw category as the evaluation category.

Reasons:

- It is the only candidate with pixel-level masks, which is required for pixel AUROC and
  for showing where the model looks.
- It has a normal-only training split, which matches the unsupervised task definition.
- Screw is the hardest category for the fixed baseline among the object categories
  (0.602), so it leaves room to show what training a specialised model adds. It also has
  five defect types and 119 anomalous test images against 41 normal ones.
- The category contains small, low-contrast defects (scratches on the head and neck,
  manipulated front, and defects on the side and top of the thread) and large appearance
  variation between individual screws, which matches the "high variance" requirement.
- Applying the baseline ranking as a selection rule avoids choosing a category that is
  already solved by a frozen backbone without any training.

The remaining limitations are stated in section 8: screw defect types are almost balanced
in count, so the imbalance in this task is the one-class split (all training images are
normal) rather than a skewed defect-type distribution.

## 3. Data preparation

- Every image is resized to 256x256 with bilinear interpolation. Masks use nearest
  neighbour interpolation and are binarised at 0.5.
- The 320 normal training images are split into 272 training and 48 validation images
  with a fixed seed (0). The official test set (41 normal, 119 anomalous) is never used
  for fitting. The only decision taken on test metrics is the convolutional patch size,
  which is documented in section 5 and listed as a limitation in section 8.5.
- Patches are extracted on a regular grid with the last row and column shifted so that
  the full image is covered. Patch size and stride are hyperparameters.
- Overlapping patch scores are averaged per pixel when the score map is assembled. The
  image score is the mean of the top 1 percent of pixels in that map. This is less
  sensitive to a single noisy patch than a maximum and still localises small defects.

## 4. Models

### 4.1 Patch feature autoencoder

A frozen ImageNet backbone (ResNet-18 or MobileNetV3-Small) converts each patch into a
feature vector (512 or 576 dimensions). Features are standardised with the training mean
and standard deviation. A symmetric MLP autoencoder reconstructs the standardised vector.
The anomaly score of a patch is the mean squared reconstruction error. The network is
trained only on patches from normal training images.

Default configuration: patch 64, stride 32, hidden 256, latent 64, dropout 0, weight
decay 1e-4, AdamW with learning rate 1e-3, cosine schedule, batch 128, at most 60 epochs,
early stopping with patience 12 on validation reconstruction loss.

The backbone is not fine-tuned. This keeps the comparison with PaDiM and PatchCore fair
and keeps training cheap.

### 4.2 Convolutional patch autoencoder

A small convolutional autoencoder works directly on RGB patches: four stride-2
convolutions reduce the patch by a factor of 16, then four transposed convolutions restore
it. For the final 128x128 patches the bottleneck is 8x8x32; for the 64x64 reference
configuration it is 4x4x32. The per-pixel squared reconstruction error forms the anomaly
map directly, so no backbone is involved. Overlapping patch maps are averaged per pixel,
and the image score is the mean of the top 1 percent of the assembled map, the same rule
as for the feature model.

Final configuration: patch 128, stride 64, latent channels 32, base channels 32, weight
decay 1e-4, AdamW with learning rate 1e-3, batch 128, at most 60 epochs, early stopping
with patience 12. Patch 64 is the reference configuration in the grid; patch 128 was
adopted because its effect is the only one that exceeds the noise of the metric, as
explained in section 5.

The two models cover the two families named in the task: a feature-space autoencoder on
top of a pretrained extractor, and a pixel-space autoencoder that learns its own features.

## 5. Hyperparameter experiments

The feature model grid has 16 runs:

- backbone: ResNet-18 and MobileNetV3-Small;
- patch size: 32, 64 and 128 with stride equal to half the patch;
- one run with 32x32 patches upscaled to 224x224 before the backbone, to test whether
  matching the pretraining resolution helps;
- latent dimension: 16, 32, 128 and 256 at patch 64 and ResNet-18;
- weight decay: 0, 1e-3 and 1e-2 at patch 64 and ResNet-18;
- dropout: 0.1 and 0.3 at patch 64 and ResNet-18.

The convolutional autoencoder grid has 11 runs:

- patch size: 32, 64 and 128;
- latent channels: 16, 64 and 128;
- weight decay: 0, 1e-3 and 1e-2;
- base width: 16 and 64.

The reference configurations were fixed before the grid: patch 64, latent 64 for the
feature model; patch 64, latent channels 32 and base width 32 for the convolutional
model. The grid moves one factor at a time around them.

A factor is changed from its reference only when its effect in the grid is larger than
the standard error of the metric, which is about 0.03 image AUROC on this test set (the
Hanley-McNeil approximation for 119 anomalous and 41 normal images). No factor qualifies
in the feature grid, so the feature model keeps its reference configuration. Patch size
does qualify for the convolutional model: 0.825, 0.828 and 0.905 image AUROC for patches
32, 64 and 128. The final convolutional model therefore uses patch 128.

Measuring that effect on the test set is a mild optimistic bias, and it is stated as a
limitation in section 8.5. Validation reconstruction loss cannot replace it, because it
rewards capacity and does not rank detectors reliably; section 8.1 quantifies this for
the feature model, where the rank correlation between validation loss and test image
AUROC is 0.30 (Spearman, p = 0.26) and the lowest validation loss does not give the best
detector.

## 6. Evaluation protocol

- Image AUROC and average precision on the 160 test images.
- Pixel AUROC and pixel average precision over all test pixels, with normal images
  contributing all-zero masks. Pixel metrics are computed on a deterministic subsample of
  two million pixels to keep memory bounded.
- Best image F1 over the test scores and the threshold that achieves it. This is an
  optimistic upper bound because it uses the labels, so it is reported only as a
  reference and is computed identically for our models and for the anomalib baselines.
- A second, calibrated F1 that uses the 99th percentile of validation normal scores as the
  threshold. This is the number a deployed system would have to use, because it does not
  look at test labels. Because the 99th percentile of 48 images is essentially the largest
  validation score, the same computation is repeated at the 80th, 85th, 90th, 95th and
  97.5th percentiles and against all 320 normal images, and stored in
  `results/calibration_sweep.csv`.
- Per-defect AUROC against normal test images, the mean score of each defect type, and the
  detection rate at the best image-F1 threshold. Detection rates use the best-F1
  threshold rather than the calibrated one so that the column stays informative for both
  models.

## 7. Baselines and profiling

PaDiM uses ResNet-18 with layers layer1, layer2 and layer3. PatchCore uses the anomalib
default WideResNet-50-2 with layer2 and layer3 and a coreset sampling ratio of 0.1. Both
are fitted on the same 320 normal training images with an 85/15 train and validation
split at seed 0; the exact validation subset differs from ours because anomalib uses its
own splitter, but no test image is used for fitting. Evaluation uses the same test set.
All metrics in the comparison are recomputed from their raw per-image scores and anomaly
maps with the same functions used for our models, so the four methods are scored
identically. The baselines keep their native image score, the maximum of the anomaly map,
and the top 1 percent variant is also recorded for the aggregation comparison.

Anomalib post-processes scores with a normalisation step that maps the validation score
range onto [0, 1] and clamps values outside it. The first run with the default settings
returned 0.54 image AUROC for PaDiM, which is close to chance, and inspection of the raw
output showed that most test scores were clamped to zero. The clamp, not the model, was
responsible, so the post-processing normalisation is disabled and the evaluator sees the
raw Mahalanobis distances and patch distances. After this change PaDiM reaches 0.72 image
AUROC, consistent with the low score of the fixed kNN baseline on this category in
section 2.2. Anomalib's own metric
dictionary is kept in the `anomalib_metrics` column for reference but is not used: it
reports exactly 0.5 image AUROC and the same image F1 for both models, which contradicts
the recomputation from the same saved scores.

Latency is measured as the median of two timed passes over 20 test images after one
untimed warm-up, including patch extraction, backbone or autoencoder forward pass, and
score map assembly. Decoding the PNG files is done once before timing and is excluded.
FLOPs are counted with `torch.utils.flop_counter.FlopCounterMode` for one patch and
multiplied by the number of patches per image. The counter adds a multiply and an add
separately, so these values are twice the multiply-accumulate numbers usually quoted in
papers. Both GPU and CPU latency are reported for the two models built in this work.

## 8. Results

Filled from the saved result files. The detailed tables are in `results/` and the figures
are in `reports/figures/`.

### 8.1 Feature autoencoder grid

Sixteen runs are reported in `results/feature_ae_grid.csv`. Three factors were varied:
the extractor and patch size, the bottleneck width, and the two regularisers. The tables
below repeat the test metrics and add the validation loss so that the two can be
compared. The final model of this family is the configuration fixed before the grid
(ResNet-18, patch 64, latent 64), shown in the first table.

Extractor and patch size, with latent 64, no dropout and weight decay 1e-4:

| extractor | patch | image AUROC | pixel AUROC | validation loss |
| --- | --- | --- | --- | --- |
| ResNet-18 | 32 | 0.671 | 0.953 | 0.0944 |
| ResNet-18 | 32, upscaled to 224 | 0.788 | 0.978 | 0.1290 |
| ResNet-18 | 64 | 0.828 | 0.914 | 0.1162 |
| ResNet-18 | 128 | 0.704 | 0.690 | 0.1280 |
| MobileNetV3-Small | 32 | 0.623 | 0.929 | 0.0659 |
| MobileNetV3-Small | 64 | 0.676 | 0.920 | 0.1106 |
| MobileNetV3-Small | 128 | 0.737 | 0.712 | 0.1185 |

Bottleneck width, with ResNet-18 and patch 64:

| latent dimension | image AUROC | pixel AUROC | validation loss |
| --- | --- | --- | --- |
| 16 | 0.818 | 0.915 | 0.1787 |
| 32 | 0.842 | 0.915 | 0.1457 |
| 64 | 0.828 | 0.914 | 0.1162 |
| 128 | 0.823 | 0.913 | 0.0816 |
| 256 | 0.814 | 0.919 | 0.0531 |

Regularisation, with ResNet-18, patch 64 and latent 64. The first four rows vary weight
decay, the last two vary dropout:

| setting | image AUROC | pixel AUROC | validation loss |
| --- | --- | --- | --- |
| weight decay 0 | 0.828 | 0.914 | 0.11619 |
| weight decay 1e-4 | 0.828 | 0.914 | 0.11624 |
| weight decay 1e-3 | 0.828 | 0.914 | 0.11619 |
| weight decay 1e-2 | 0.825 | 0.913 | 0.11616 |
| dropout 0.1 | 0.825 | 0.912 | 0.14316 |
| dropout 0.3 | 0.801 | 0.907 | 0.19644 |

The extractor and the patch size dominate. ResNet-18 at patch 64 gives 0.828 image AUROC
and 0.914 pixel AUROC. MobileNetV3-Small is worse at patches 32 and 64 and better at 128,
but it is much cheaper, which is why it was kept in the grid as a cost reference. Patch
128 collapses pixel AUROC to 0.69 because a 128x128 patch is a coarse localisation unit:
the assembled map contains only 3x3 patch centres. Patch 32 localises best without
upscaling (0.953) but its image score is noisier. Upscaling those patches to 224 pixels
before the backbone recovers image AUROC to 0.788 and raises pixel AUROC to 0.978, at
roughly 56 times the backbone compute of the patch-64 model (225 patches per image
instead of 49, each 12.25 times larger in area).

Bottleneck width has almost no effect between 16 and 256. The best image AUROC in the
whole grid is 0.842 at latent 32, but the five patch-64 variants span 0.028 AUROC, which
is about the standard error of a single AUROC estimate on this test set (0.032 by the
Hanley-McNeil approximation). Weight
decay is inert: all four values land within 0.003 AUROC. Dropout only hurts, and at 0.3
the image AUROC falls to 0.801.

Validation loss ranks configurations poorly. Across the 16 runs its Spearman correlation
with test image AUROC is 0.30 (p = 0.26). The two lowest validation losses belong to
latent 256 (0.0531) and MobileNetV3 at patch 32 (0.0659), whose image AUROC is 0.814 and
0.623. A wider bottleneck reconstructs normal patches more accurately without learning
anything more discriminative, and the MobileNet features are simply easier to
reconstruct. This is why the final model is the configuration fixed before the grid
rather than the validation-loss winner.

### 8.2 Convolutional autoencoder grid

Eleven runs are reported in `results/cae_grid.csv`. Each row changes one factor from the
patch-64 reference configuration. All runs used the full 60 epochs: the best validation
loss always fell within the last twelve epochs, so early stopping never triggered.

| patch | latent channels | base channels | weight decay | image AUROC | pixel AUROC | validation loss |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 32 | 1e-4 | 0.825 | 0.929 | 0.000089 |
| 64 | 32 | 32 | 1e-4 | 0.828 | 0.915 | 0.000133 |
| 128 | 32 | 32 | 1e-4 | 0.905 | 0.912 | 0.000280 |
| 64 | 16 | 32 | 1e-4 | 0.835 | 0.920 | 0.000189 |
| 64 | 64 | 32 | 1e-4 | 0.868 | 0.914 | 0.000110 |
| 64 | 128 | 32 | 1e-4 | 0.863 | 0.916 | 0.000105 |
| 64 | 32 | 16 | 1e-4 | 0.846 | 0.912 | 0.000157 |
| 64 | 32 | 64 | 1e-4 | 0.827 | 0.919 | 0.000122 |
| 64 | 32 | 32 | 0 | 0.826 | 0.916 | 0.000134 |
| 64 | 32 | 32 | 1e-3 | 0.834 | 0.915 | 0.000134 |
| 64 | 32 | 32 | 1e-2 | 0.830 | 0.915 | 0.000133 |

Patch size is the only factor with an effect larger than the standard error. Moving from
patch 64 to patch 128 adds 0.077 image AUROC, about 2.4 standard errors, and also gives
the best image AUPR (0.964) and image F1 (0.918). The mechanism is visible in the
localisation metrics: pixel AUROC falls slightly, from 0.915 to 0.912, because a 128x128
patch is averaged over a coarser grid, while the image score becomes much more reliable.
Smaller patches reconstruct small defects too well, so the image score reacts less.

The bottleneck is ambiguous. Latent 64 reaches 0.868 and latent 128 reaches 0.863, about
0.04 above the reference, but the trend is not monotone (latent 16 gives 0.835) and the
gain is about one standard error. Base width behaves the same way: 16 channels give 0.846,
64 channels give 0.827, and 32 gives 0.828, with no consistent direction. Weight decay is
inert again, with all four values inside 0.008 AUROC.

Validation loss repeats the failure seen in section 8.1. The lowest value belongs to
latent 128 (0.000105), which reaches 0.863, while the selected patch-128 model has the
worst validation loss of the grid (0.000280) and the best detection quality (0.905).
Larger patches are harder to reconstruct, and lower reconstruction error is not the same
as better anomaly separation.

The final convolutional model (patch 128) reaches 0.904 image AUROC, 0.918 image F1,
0.964 image AUPR, 0.912 pixel AUROC and 0.066 pixel F1 on the test set, matching its grid
row within the run-to-run variation of about 0.003 AUROC.

### 8.3 Final comparison

Table 2 collects the test metrics of the two models built here and the two anomalib
baselines. Every number is recomputed from saved per-image scores and anomaly maps with
the same functions. Image F1 is the best F1 over all thresholds, so it is an optimistic
upper bound for all four models. Pixel AUPR is computed over all test pixels, where only
0.25 percent of pixels are defective, so its chance level is 0.0025 rather than 0.5.

| model | image AUROC | image AUPR | image F1 | pixel AUROC | pixel AUPR | pixel F1 |
| --- | --- | --- | --- | --- | --- | --- |
| feature autoencoder | 0.828 | 0.930 | 0.883 | 0.914 | 0.026 | 0.075 |
| convolutional autoencoder | 0.904 | 0.964 | 0.918 | 0.912 | 0.028 | 0.066 |
| PaDiM | 0.720 | 0.835 | 0.874 | 0.977 | 0.115 | 0.194 |
| PatchCore | 0.947 | 0.981 | 0.934 | 0.988 | 0.329 | 0.370 |

PatchCore leads every column, which is the expected result for the strongest published
method on MVTec AD and a useful sanity check on the pipeline. Both trained models beat
PaDiM clearly on image AUROC, by 0.11 and 0.18, and both lose to it on pixel AUROC by
0.06. The split between image and pixel behaviour is the main qualitative finding: a
compact patch autoencoder can rank whole images well while its pixel map stays coarse.

The pixel average precision makes the localisation gap much larger than pixel AUROC
suggests: 0.026 and 0.028 for the two models against 0.115 for PaDiM and 0.329 for
PatchCore. A pixel AUROC of 0.91 with an average precision of 0.03 means the map ranks a
defect pixel above a background pixel most of the time, but the defect pixels are rarely
at the very top of the ranking, because the reconstruction error is also elevated on the
rest of the object. Figure 16 shows this directly for the convolutional model: the whole
screw body is warm and the defect is only slightly warmer. The feature model has the same
weakness for a different reason: its score map is assembled from 32-pixel blocks, so a
small scratch can only raise a block that also contains clean metal.

Table 3 replaces the best-F1 threshold with thresholds derived only from normal images, so
these numbers are what a deployed system could reach without test labels.

| model | normal set | quantile | threshold | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| feature autoencoder | validation, 48 | 0.80 | 0.403 | 0.790 | 0.912 | 0.697 |
| feature autoencoder | validation, 48 | 0.99 | 0.530 | 0.408 | 0.939 | 0.261 |
| feature autoencoder | all, 320 | 0.99 | 0.486 | 0.506 | 0.953 | 0.345 |
| convolutional autoencoder | validation, 48 | 0.80 | 0.00879 | 0.513 | 1.000 | 0.345 |
| convolutional autoencoder | validation, 48 | 0.99 | 0.01182 | 0.000 | 0.000 | 0.000 |
| convolutional autoencoder | all, 320 | 0.99 | 0.01220 | 0.000 | 0.000 | 0.000 |

The feature model calibrates in the usual sense. The mean normal score is 0.355 on the 48
validation images and 0.351 on the 41 normal test images, so the two distributions nearly
coincide, and the 99th percentile of validation scores gives 0.408 test F1 with 4.9
percent false positives. Using all 320 normal images moves the threshold down to 0.486
and improves F1 to 0.506, because the training half of the normals has a lower mean than
the validation half.

The convolutional model fails to calibrate, and the failure is systematic rather than a
tuning mistake. Its mean normal score is 0.0080 on validation but 0.0066 on the normal test
images, 17 percent lower, and the largest anomalous test score (0.0116) is below the
validation 99th percentile (0.0118), so nothing is flagged. The threshold that maximises
test F1, 0.00676, sits at the 10th percentile of the validation normal scores. No quantile
at or above 0.8 can recover it, and using the training normals makes the threshold higher
still, because their mean (0.0079) is also above the test normal mean. The cause is that
the convolutional score is a raw reconstruction error whose level follows the overall
appearance of the image, and the normal test screws are systematically easier to
reconstruct than the normal training screws. The feature model is less exposed to this
because its score is a reconstruction error in standardised feature space, which removes
much of the image-level variation before the autoencoder sees it.

Figure 15 shows the full sweep. Neither model has a quantile that is simultaneously safe
and sensitive, but the shape differs: the feature model degrades gradually and keeps an
F1 between 0.4 and 0.5 at the 99th percentile, while the convolutional model falls below
0.03 by the 95th percentile and reaches zero at 0.975.

Table 4 breaks the image AUROC down by defect type.

| defect | images | feature AUROC | conv AUROC | feature detection | conv detection |
| --- | --- | --- | --- | --- | --- |
| manipulated_front | 24 | 0.710 | 0.852 | 0.792 | 0.958 |
| scratch_head | 24 | 0.947 | 0.936 | 1.000 | 1.000 |
| scratch_neck | 25 | 0.906 | 0.875 | 0.920 | 1.000 |
| thread_side | 23 | 0.769 | 0.887 | 0.870 | 0.957 |
| thread_top | 23 | 0.801 | 0.972 | 0.870 | 1.000 |

Detection is the fraction of that defect type above the best image-F1 threshold of the
model. Manipulated_front is the hardest type for both models: it changes the shape of the
tip, and rotation or small pose changes produce a similar image-level difference, so the
score is high for many normal screws too. The convolutional model is much better on
thread_top and thread_side, which lie on the thread rather than on the head, while the
feature model is slightly better on scratch_neck. At the best threshold the convolutional
model detects at least 95.7 percent of every defect type; the feature model misses about
one in five manipulated_front and thread_side images.

Table 5 repeats the image AUROC for five ways of turning the score map into one number.

| model | mean | maximum | top 5 percent | top 1 percent | top 0.5 percent |
| --- | --- | --- | --- | --- | --- |
| feature autoencoder | 0.790 | 0.828 | 0.826 | 0.828 | 0.828 |
| convolutional autoencoder | 0.861 | 0.794 | 0.901 | 0.904 | 0.901 |

The top 1 percent rule is the right default. For the convolutional model the maximum loses
0.110 AUROC because one noisy pixel can decide the image score, while the top 1 percent
averages over 655 pixels and is stable. For the feature model the maximum, top 1, top 0.5
and top 0.1 percent all give the same 0.828, because each patch covers 32x32 unique pixels,
or 1.6 percent of the image, so the top 1 percent already lies inside the hottest patch.
The baselines are almost insensitive to the rule: PaDiM scores 0.720 with its native
maximum and 0.724 with the top 1 percent, PatchCore 0.947 and 0.949. Since the difference
is negligible, the comparison in Table 2 does not depend on which aggregation each method
prefers.

### 8.4 Profiling

The numbers come from `results/profile.csv` and use the two final checkpoints. Latency is
per 256x256 image on the GTX 1060 and on the 12-thread CPU. FLOPs are counted with the
multiply and add convention of `FlopCounterMode`, so halve them to compare with the usual
MAC figures.

| model | GPU latency (ms) | CPU latency (ms) | GFLOP per image | image AUROC |
| --- | --- | --- | --- | --- |
| feature autoencoder | 12.8 | 42.9 | 14.54 | 0.828 |
| convolutional autoencoder | 3.9 | 10.0 | 2.79 | 0.904 |

The convolutional model is 3.3 times faster on the GPU, 4.3 times faster on the CPU and
5.2 times cheaper in FLOPs, while scoring 0.076 higher image AUROC, so it wins on both
axes of the comparison and is the model to prefer when latency matters. The GPU speed-up
over the CPU is 3.4 times for the feature model and only 2.6 times for the convolutional
model, because the convolutional model processes nine larger patches per image and spends
a larger share of its time in the per-image Python loop that assembles the score map.

Almost all of the feature model's cost is the frozen extractor: one ResNet-18 pass on a
64x64 patch is 0.296 GFLOP and the MLP autoencoder adds 0.001 GFLOP, so the backbone
accounts for 99.7 percent of the per-patch compute. The model uses 49 patches per image.
The convolutional autoencoder is 0.310 GFLOP per 128x128 patch and uses 9 patches per
image. Both models use fewer than half a million parameters in the part that is trained:
296 thousand for the feature autoencoder and 463 thousand for the convolutional one. The
frozen ResNet-18 adds 11.2 million parameters that are never updated.

It is worth noting that the feature model at patch 32 upscaled to 224, the configuration
that reached 0.788 image AUROC, would cost 225 patches of 224x224, about 0.8 TFLOP per
image by the same counting convention, roughly 56 times the profiling cost of the
feature model that was actually selected. That configuration was rejected in section 8.1
for this reason as well as for its lower image AUROC.

### 8.5 Discussion points

The task asked for a comparison across model families, a hyperparameter study, and latency
and compute numbers. The results support four conclusions.

First, a small convolutional autoencoder trained from scratch is competitive with a
feature-space autoencoder built on a pretrained backbone. It reaches 0.904 image AUROC
against 0.828, while using 2.8 GFLOPs against 14.5 and 3.9 ms against 12.8 ms per image on
the GPU. The reason is visible in the grids: the frozen features at patch 64 already
compress a 64x64 patch into 512 numbers, and the MLP autoencoder only has to invert that
code; it never sees the pixels, so it cannot notice small scratches that survive
pooling. The convolutional model sees the raw patch and its reconstruction error is
spatially resolved, which helps on thread_top and thread_side.

Second, none of the trained models localises as well as the baselines. Pixel AUROC is
close (0.91 against 0.98) but pixel average precision is four to twelve times lower. The
practical reading is that the models could drive an image-level reject decision, but the
heat maps are not accurate enough to crop the defect for a human inspector without further
work.

Third, calibration is the weakest part of the pipeline. The feature model keeps a useful
F1 at a threshold taken from validation normals (0.41 at the 99th percentile, 0.79 at the
80th), but the convolutional model cannot be operated without test labels at any quantile
above 0.8. This is not only a small-sample problem: 48 validation images are enough to
estimate a 99th percentile only crudely, but the failure persists when all 320 normal
images are used. The cause is a shift in the normal score level between the training and
test splits, and it would need either per-image normalisation or a calibration set drawn
from the deployment distribution.

Fourth, validation reconstruction loss is not a model-selection criterion. The Spearman
correlation with test image AUROC is 0.30 over the feature grid, the lowest validation loss
there belongs to latent 256 (0.0531, image AUROC 0.814), and in the convolutional grid the
lowest validation loss belongs to latent 128 (0.000105, image AUROC 0.863) while the
selected patch-128 model has the worst validation loss of its grid (0.000280) and the best
image AUROC (0.905). Reconstruction and discrimination are different objectives: a model
that reconstructs normal patches better also reconstructs small defects better.

A cheap post-hoc check suggests where the remaining headroom is. The two models make
different errors: their image scores correlate at 0.29, and their per-defect AUROCs differ
most on thread_top (0.801 against 0.972) and manipulated_front (0.710 against 0.852).
Averaging the two scores after standardising each with the mean and standard deviation of
its own validation normals gives 0.918 image AUROC, above either model and within one
standard error of PatchCore (0.947), with a calibrated F1 of 0.455 at the 99th percentile
of the fused validation scores and no false positives. This is a post-hoc observation on the test set, not a
validated model, and it is stored in `results/fusion_check.csv`; it is mentioned because it
costs nothing and it shows that the two families fail differently.

Limitations that should be stated in the report:

- One category and one seed for the final models. The standard error of image AUROC on 119
  anomalous and 41 normal images is about 0.032, so the 0.828 to 0.904 gap between the two
  trained models is about 2.4 standard errors, and the 0.720 to 0.828 gap over PaDiM is
  about 3.4. Differences below 0.03 should not be interpreted.
- The convolutional patch size was selected on the test metric. The gain of 0.077 is
  larger than the standard error, but the selection makes the reported 0.904 mildly
  optimistic. The grid run and the final run agree to 0.001, so the run-to-run noise
  (about 0.003) is not the issue; the issue is that the same test set was used for the
  decision and for the report.
- The baselines use anomalib defaults: ResNet-18 for PaDiM, WideResNet-50-2 for PatchCore,
  no tuning of layers, input size or coreset ratio. A tuned PaDiM could plausibly close
  part of its image-level gap.
- Images are resized from 1024x1024 to 256x256 before any processing. Small screw defects
  lose pixel detail, which penalises all four methods and probably the localisation metrics
  most.
- Pixel average precision is computed on a deterministic two-million-pixel subsample, and
  the positive rate is 0.25 percent, so the average precision is dominated by a small
  number of high-ranked defect pixels even though pixel AUROC is stable.
- The 48-image validation split is too small for a 99th-percentile threshold; the sweep in
  section 8.3 shows the whole range instead of hiding this behind one number.

## 9. Figures

Sixteen figures are in `reports/figures/`. Figures 1 and 2 are written by
`anomaly-lab screen`, figure 3 by `anomaly-lab samples`, and figures 4 to 16 by
`anomaly-lab figures`. The captions below are written to be pasted under the images.

| file | what it shows | caption |
| --- | --- | --- |
| `fig01_dataset_screening.png` | four panels: category sizes, anomalous fraction per candidate, kNN baseline difficulty, defect-type imbalance against mask area | Candidate datasets. (a) Number of normal training and anomalous test images in each MVTec AD category. (b) Fraction of anomalous images in the test split of every candidate. (c) Image AUROC of the fixed kNN baseline on ResNet-18 patch features; the dashed line marks 0.8. (d) Defect-type imbalance against mean ground-truth mask area for MVTec AD, object categories in blue and texture categories in yellow; marker size follows the standard deviation of the mask area. Screw combines very small masks with the lowest baseline score. |
| `fig02_category_ranking.png` | the 15 MVTec categories and casting, sorted by kNN AUROC, screw highlighted | Image AUROC of the fixed kNN baseline on normal patch features for every candidate. The selected category, screw, is shown in red. It is the only object category below 0.65, which leaves the most room for a trained model. |
| `fig03_dataset_samples.png` | one normal screw and one image per defect type with the mask in red | Example test images of the screw category. The top-left image is normal; the others show one example of each defect type with the ground-truth mask in red. The defects are small relative to the image and several of them differ from normal mainly by a local change in shape. |
| `fig04_score_heatmaps.png` | image, mask, feature-autoencoder map and overlay for one image per defect type | Anomaly maps of the feature autoencoder on one test image per defect type. Columns are the input, the ground-truth mask, the assembled score map and the overlay. The maps are blocky because each 64x64 patch contributes only a 32x32 region of unique pixels after overlapping patches are averaged. |
| `fig05_training_curves.png` | train and validation reconstruction loss for the two final models, log scale, best epoch marked | Reconstruction loss during training of the two final models. Both curves are on a logarithmic scale; the dotted line marks the epoch with the lowest validation loss, which is also the checkpoint that was kept. Neither model triggered early stopping. |
| `fig06_patch_backbone.png` | image and pixel AUROC against patch size for ResNet-18 and MobileNetV3-Small | Effect of patch size and extractor on the feature autoencoder. (a) Image AUROC. (b) Pixel AUROC. ResNet-18 at patch 64 is the best image-level configuration, and patch 128 collapses localisation because the assembled map becomes a 3x3 grid. |
| `fig07_regularization.png` | three panels for latent width, weight decay and dropout, common y axis | Regularisation study of the feature autoencoder, all panels on the same scale. (a) Bottleneck width. (b) Weight decay. (c) Dropout. Weight decay is inert, dropout only hurts, and the bottleneck width moves the metric by less than its standard error. |
| `fig08_cae_grid.png` | four panels for patch size, latent channels, weight decay and base width | Hyperparameter study of the convolutional autoencoder. (a) Patch size, the only factor with an effect larger than the standard error. (b) Latent channels. (c) Weight decay. (d) Base width. Pixel AUROC falls as patches grow while image AUROC rises. |
| `fig09_score_distribution.png` | test score histograms for both models with the calibrated threshold | Score separation on the test set for the feature autoencoder (left) and the convolutional autoencoder (right). The dashed line is the 99th percentile of the validation normal scores. For the convolutional model this threshold lies beyond every test score, so the calibrated classifier flags nothing. |
| `fig10_roc.png` | image-level ROC curves for the four models | Image-level ROC curves on the test set. PatchCore dominates, the convolutional autoencoder is second, and PaDiM is below both trained models despite its strong pixel-level score. |
| `fig11_model_comparison.png` | grouped bars of image AUROC, pixel AUROC and best F1 | Comparison of the two trained models with PaDiM and PatchCore. The y axis starts at 0.4 to make the differences visible. Image F1 is the best F1 over thresholds for every model. |
| `fig12_latency_flops.png` | GPU latency, CPU latency and GFLOPs per image for the two models | Cost of the two models on a GTX 1060 and a 12-thread CPU. (a) Latency on the GPU. (b) Latency on the CPU. (c) FLOPs per image, counted with one multiply and one add as two operations. The convolutional model is cheaper on every axis and more accurate. |
| `fig13_validation_gap.png` | validation loss against test image AUROC for the 16 feature runs | Validation reconstruction loss against test image AUROC for the 16 feature configurations. Colour is the backbone and marker size grows with patch size. The relationship is weak (Spearman 0.30) and the lowest validation losses do not belong to the best detectors. |
| `fig14_per_defect_scores.png` | mean image score per defect type with standard deviation, both models | Mean image score of each defect type with one standard deviation, for the feature autoencoder (left) and the convolutional autoencoder (right). The dashed line is the mean normal score. Manipulated_front is closest to normal for both models. |
| `fig15_calibration.png` | test F1 against the quantile of the normal scores used as the threshold | Test F1 as a function of the quantile of the normal score distribution used as the threshold. The blue curve uses all 320 normal images and the red curve the 48 validation images. The feature model degrades gradually and keeps an F1 between 0.4 and 0.5 at the 99th percentile; the convolutional model falls below 0.03 by the 95th percentile and reaches zero at 0.975, because its normal scores on the test set are lower than on the training set. |
| `fig16_cae_score_heatmaps.png` | the same layout as figure 4 for the convolutional autoencoder | Anomaly maps of the convolutional autoencoder on the same test images as figure 4. The colour scale is clipped at the 99.5th percentile of each map so that the structure is visible. The error is elevated along the whole object, not only on the defect, which explains the low pixel average precision despite the high pixel AUROC. |

## 10. Reproduction

Run from the repository root. The order matters only in that `evaluate` needs the two
checkpoints and `figures` needs the files written by `evaluate` and `benchmark`.

```bash
uv sync --extra benchmark
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

Notes on the commands:

- `train-feature` and `train-cae` run the whole grid, not a single model. They append to
  their CSV and history files after every finished configuration and accept `--resume`, so
  an interrupted grid continues from the last saved configuration.
- `train-feature-final` uses the configuration fixed in section 5 and writes
  `checkpoints/feature_ae_best.pt`. `train-cae-final` needs the explicit patch and stride
  flags shown above, because its defaults are the reference configuration rather than the
  selected one; it writes `checkpoints/cae_best.pt`.
- `benchmark` needs the optional dependency group. The allocator flag in the command above
  avoids fragmentation on a 6 GB card; the command uses batch 8 for PaDiM and batch 4 for
  PatchCore to stay inside 6 GB.
- `screen` needs the casting and PCB datasets in `data/casting` and `data/pcb_aoi` as well
  as MVTec AD in `data/mvtec_ad`. The rest of the pipeline needs only MVTec AD.

Raw data and checkpoints are not committed. MVTec AD must be placed in `data/mvtec_ad`
with the original category folder layout.
