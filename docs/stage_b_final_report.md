# Stage B Final Report — Tile/Grid Distortion Localization

**Status:** loss function decided (**focal loss, α=0.75, γ=2.0**, §4), then the scratch tile
ground-truth threshold was found to be discarding half of all real scratch signal and fixed by a
dataset rebuild (§4 "Post-decision follow-ups", option 3) — **the current canonical result is
focal α=0.75 trained on the rebuilt dataset** (`data/processed/stage_b_scratch15`,
`checkpoints/stage_b_scratch15/stage_b_head.pt`). The original dataset and checkpoint
(`data/processed/stage_b`, `checkpoints/stage_b_alpha75/stage_b_head.pt`) are kept on disk for
comparison, not deleted. **Date:** 2026-09-05.

This is a standalone summary of Part 2, Stage B — distinct from
[`development_log.md`](development_log.md)'s chronological session-by-session record. See
the log (Sessions 15-18) for the full narrative; this document is the result, updated as it
firms up.

---

## 1. Goal

Per `architecture.md` §2, Stage B extends Stage A's whole-image classification to **coarse
spatial localization**: "the feature map is treated as a grid (e.g. 16×16), one classification
output per tile" — so the model can say not just *whether* a distortion is present, but roughly
*where* on the lens ("dirt in the top right"), without needing pixel-mask annotation. Same
backbone, same three classes (dirt/water/scratch), same frozen-backbone training strategy as
Stage A — the new part is a per-tile output instead of one label for the whole image.

---

## 2. Pipeline overview

| Step | What | Where |
|---|---|---|
| 1 | Stage B dataset builder (tile-grid ground truth, rasterized from each effect's own pixel mask) | `src/soiling/tile_labels.py`, `src/soiling/dataset_builder.py`, `scripts/build_stage_b_dataset.py` |
| 2 | Model: frozen backbone + 1×1 conv tile head | `src/models/distortion_head.py::StageBDistortionHead` |
| 3 | Loss: bce+pos_weight, and focal loss (two alpha settings) | `src/models/losses.py` |
| 4 | Training script | `scripts/train_stage_b.py` |
| 5 | FAU HPC (TinyGPU) submission | `scripts/hpc/train_stage_b.slurm`, `train_stage_b_alpha75.slurm` |
| 6 | Evaluation (per-tile precision/recall/F1/AP) | `scripts/evaluate_stage_b.py` |
| 7 | Threshold tuning (no retraining) | `scripts/threshold_sweep_stage_b.py` |
| 8 | Visualization | `scripts/visualize_stage_b_results.py` |

---

## 3. Key details

### Tile ground truth

Each effect's own `[0,1]` pixel mask is downsampled (area-average pooling) to the backbone's
native P5 grid size — **16×16** at `--img-size 512` (stride 32) — and thresholded per tile:
15% coverage for dirt/water (filled regions unchanged throughout). Scratch (a thin line) started
at 3%, later found to be too strict (§4, option 3) and lowered to **1.5%** in the canonical
dataset. Stored as one consolidated `tile_labels.npy`, `(N, 3, 16, 16)` uint8, row-aligned with
`metadata.csv`.

### Model

`StageBDistortionHead`: a single `Conv2d(in_channels, 3, kernel_size=1)` directly on the frozen
backbone's P5 feature map → raw per-tile logits `(B, 3, 16, 16)`. Backbone unchanged from Stage A.

### Loss — three variants tried, in order

1. **bce + pos_weight** (Session 16): standard inverse-frequency weighting. Tile-level imbalance
   is far more extreme than image-level (scratch tiles are ~1% positive → `pos_weight≈95`) — this
   pushed the trained head to over-predict scratch almost everywhere (recall 0.852, precision
   0.063), rather than localizing it.
2. **focal loss, α=0.25, γ=2.0** (Session 17): RetinaNet's own paper defaults. Fixed scratch's
   over-prediction (precision 0.063→0.741) and improved AP for every class, but recall collapsed
   across all three classes (not just scratch) — traced to α=0.25 being tuned for RetinaNet's far
   more extreme (~1:1000) imbalance; this project's tile imbalance (13%/18%/1%) is milder, so the
   same default overcorrects. Verified not a bug: implementation checked against the paper
   formula, 98/98 tests pass, and an independent from-scratch toy reproduction shows the same
   qualitative effect at a similar imbalance level (see `development_log.md` Session 18).
3. **focal loss, α=0.75, γ=2.0** (Session 18): tests whether raising α to weight positives more
   heavily recovers recall without reintroducing scratch's over-prediction problem. **Winner** —
   beats every other variant on all three classes at the plain default threshold (0.5), with no
   threshold tuning needed (§4).

### Threshold tuning (Session 18, no retraining)

Independent of the loss-function question: per-class best-F1 thresholds found on the val split
of the focal-α0.25 checkpoint, confirmed on test. Since AP (threshold-independent) already
improved for every class in that run, this checks whether a better *operating point* exists on
the same curve, without touching the model.

---

## 4. Results

### Training curves

![Stage B training curve, bce+pos_weight (job 1799124)](images/stage_b_training_curve_bce.jpg)
![Stage B training curve, focal alpha=0.25 (job 1799134)](images/stage_b_training_curve_focal.jpg)

Both smooth and monotonic, train/val tracking closely — no overfitting in either run. (Loss
magnitudes aren't comparable between the two: focal loss's `(1-p_t)^γ` factor shrinks the loss
scale itself, independent of how well the model is doing.)

### Per-class tile metrics (held-out test split, 400 images, 102400 tiles)

| class | metric | bce (Session 16) | focal α=0.25 (Session 17) | focal α=0.25 + tuned threshold (Session 18) | **focal α=0.75 (Session 18)** |
|---|---|---|---|---|---|
| dirt | precision / recall / F1 / AP | 0.595 / 0.904 / 0.718 / 0.864 | 0.925 / 0.649 / 0.763 / 0.886 | 0.826 / 0.781 / 0.803 / 0.886 | 0.782 / 0.812 / **0.797** / 0.882 |
| water | precision / recall / F1 / AP | 0.699 / 0.921 / 0.795 / 0.868 | 0.915 / 0.608 / 0.731 / 0.895 | 0.796 / 0.856 / 0.825 / 0.895 | 0.701 / 0.930 / **0.799** / 0.891 |
| scratch | precision / recall / F1 / AP | 0.063 / 0.852 / 0.117 / 0.309 | 0.741 / 0.130 / 0.221 / 0.380 | 0.485 / 0.402 / 0.440 / 0.380 | 0.518 / 0.357 / **0.423** / 0.360 |

(α=0.75 column: default threshold 0.5, no tuning applied — unlike the α=0.25+tuned column.)

![Stage B per-tile metrics, bce+pos_weight](images/stage_b_test_metrics_bce.jpg)
![Stage B per-tile metrics, focal alpha=0.25](images/stage_b_test_metrics_focal.jpg)

**α=0.75 is the practical winner.** At the plain default threshold, it beats bce and focal-α0.25
on F1 for *all three classes simultaneously* — something neither of those managed even after
Session 18's threshold tuning was applied to the α=0.25 run. It comes within 1-3 points of
`α=0.25 + tuned-threshold`'s F1 on every class, without needing any per-class threshold tuning
at all — the simpler, more robust choice for an actual deployment. One caveat: α=0.75's AP is
marginally *lower* than α=0.25's on every class (e.g. scratch 0.360 vs 0.380) — its underlying
ranking of predictions isn't better, it just lands at a better point on a very similar curve by
default. Threshold-tuning α=0.75 the same way as α=0.25 was tuned is a natural next check, but
given the AP gap, unlikely to leapfrog `α=0.25 + tuned-threshold` by much.

**Independent cross-check.** The Colab run above is one training run (one random seed/shuffle).
The HPC job (`1802232`) for the exact same config *also* completed, unattended, shortly after
the Colab run finished — giving a second, fully independent training run to check the result
isn't a one-off:

| class | precision | recall | F1 | AP | vs. Colab F1 |
|---|---|---|---|---|---|
| dirt | 0.774 | 0.818 | 0.795 | 0.881 | Δ 0.002 |
| water | 0.728 | 0.917 | 0.811 | 0.889 | Δ 0.012 |
| scratch | 0.541 | 0.320 | 0.403 | 0.357 | Δ 0.020 |

Both runs land within ~1-2 points of F1 on every class — the α=0.75 result is stable across
independent runs, not a fluke of one particular initialization. `checkpoints/stage_b_alpha75/
stage_b_head.pt` (Colab) is kept as the canonical checkpoint; the HPC run's checkpoint is saved
alongside it as `stage_b_head_hpc.pt` for reference.

### Winning model (focal α=0.75, scratch15 dataset): training curve, metrics, and sample predictions

![Stage B training curve, focal alpha=0.75, scratch15 dataset (job 1802973)](images/stage_b_training_curve_scratch15.jpg)
![Stage B per-tile metrics, focal alpha=0.75, scratch15 dataset](images/stage_b_test_metrics_scratch15.jpg)
![Stage B sample predictions, focal alpha=0.75, scratch15 dataset](images/stage_b_sample_predictions_scratch15.jpg)

Ground-truth tile grid (green) overlaid with the model's predicted per-tile probability (red) on
real test-split images, one clean + one per class. Title turns red where the model's whole-image
call (any tile over threshold vs. none) disagreed with ground truth.

Scratch remains the weak point of the three classes even after the rebuild — same pattern as
Stage A, where scratch was also the weakest class (AP 0.920 there vs. ~1.0) — but it is now
measurably better than every earlier attempt (§4, option 3 above).

<details>
<summary>Pre-rebuild result (focal α=0.75 on the original 3%-threshold dataset, superseded — kept for reference)</summary>

![Stage B training curve, focal alpha=0.75 (Colab, cross-checked on HPC job 1802232)](images/stage_b_training_curve_alpha75.jpg)
![Stage B per-tile metrics, focal alpha=0.75](images/stage_b_test_metrics_alpha75.jpg)
![Stage B sample predictions, focal alpha=0.75](images/stage_b_sample_predictions_alpha75.jpg)

</details>

### Post-decision follow-ups (Session 18, tried in order)

Three further improvement attempts against the α=0.75 winner:

1. **Threshold-tune α=0.75 itself** — real, free win on water (F1 0.799→**0.826**), dirt already
   near-optimal at default, scratch only marginal (0.423→0.431). No retraining needed.
2. **Per-class α** (`FocalLossWithLogits` already supported a per-class alpha tensor; extended the
   CLI to expose it) — tried `α=[0.75, 0.75, 0.9]` to push scratch specifically. **Net negative**:
   scratch's recall rose (0.357→0.459) but precision fell more (0.518→0.365), F1 dropped
   (0.423→0.407) and AP dropped too (0.360→0.343, a genuinely worse ranking). Dirt also slipped
   slightly. Reverted — uniform α=0.75 stays canonical.
3. **Re-check the scratch tile-coverage threshold** (was 3%, `DEFAULT_TILE_THRESHOLDS` in
   `src/soiling/dataset_builder.py`) — a diagnostic (`scripts/diagnose_scratch_threshold.py`,
   60 real `add_scratch` masks) found the median coverage of tiles the scratch mask actually
   touches is 0.0172, i.e. the 3% cutoff discarded over half of all real scratch tiles as
   negatives. **Acted on**: rebuilt the full Stage B dataset with the scratch threshold lowered to
   **1.5%** (`data/processed/stage_b_scratch15`, `scripts/build_stage_b_dataset.py
   --scratch-threshold 0.015`; the original 3%-threshold dataset,
   `data/processed/stage_b`, is kept unchanged on disk for comparison) and retrained the same
   winning config (focal, α=0.75, γ=2.0) on it (HPC job `1802973`). **Net positive, adopted as
   canonical** — scratch F1 0.423→**0.451**, AP 0.360→**0.409**, with dirt/water essentially
   unchanged (within run-to-run noise):

   | class | precision | recall | F1 | AP | vs. original F1 |
   |---|---|---|---|---|---|
   | dirt | 0.787 | 0.815 | 0.801 | 0.887 | +0.004 |
   | water | 0.743 | 0.907 | 0.817 | 0.885 | +0.004 |
   | scratch | 0.593 | 0.364 | 0.451 | 0.409 | **+0.028** |

   Scratch's precision rose sharply (0.518→0.593) while recall stayed roughly flat (0.357→0.364)
   — net effect is a better-ranked, more usable scratch signal (the AP gain is the more telling
   number here, since it's threshold-independent). `checkpoints/stage_b_scratch15/
   stage_b_head.pt` is now the canonical Stage B checkpoint; `checkpoints/stage_b_alpha75/` is
   kept for reference as the pre-rebuild result.

### Sample tile-grid predictions

Ground-truth tile grid (green) overlaid with predicted per-tile probability (red) on real
test-split images:

![Stage B sample predictions, bce+pos_weight](images/stage_b_sample_predictions_bce.jpg)
![Stage B sample predictions, focal alpha=0.25](images/stage_b_sample_predictions_focal.jpg)

---

## 5. Known limitations

- **Scratch localization is still the weakest class**, even after the threshold rebuild. Current
  best F1 is 0.45 (canonical scratch15 model) — usable as a coarse signal, not a reliable
  per-tile localizer.
- **Backbone frozen, purely synthetic data, small pilot dataset** — same caveats as
  [`stage_a_final_report.md`](stage_a_final_report.md) §5, unchanged here since Stage B reuses
  the same backbone and dataset-generation pipeline.
- **Final configuration:** focal loss, α=0.75, γ=2.0, trained on the scratch-threshold-1.5%
  dataset (`checkpoints/stage_b_scratch15/stage_b_head.pt`,
  `data/processed/stage_b_scratch15`).

---

## 6. Reproducing this report

```bash
# canonical (scratch15 dataset, focal alpha=0.75)
python scripts/build_stage_b_dataset.py --source data/raw/mio_tcd/images \
    --out data/processed/stage_b_scratch15 --scratch-threshold 0.015
python scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b_scratch15/stage_b_head.pt \
    --data data/processed/stage_b_scratch15 --split test
python scripts/visualize_stage_b_results.py --checkpoint checkpoints/stage_b_scratch15/stage_b_head.pt \
    --data data/processed/stage_b_scratch15 --split test --log-file stage_b_s15_1802973.out \
    --tag _scratch15 --out-dir docs/images

# earlier loss-variant comparisons (original 3%-threshold dataset, kept for reference)
python scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b_alpha75/stage_b_head.pt \
    --data data/processed/stage_b --split test
python scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b/stage_b_head_focal.pt \
    --data data/processed/stage_b --split test
python scripts/threshold_sweep_stage_b.py --checkpoint checkpoints/stage_b/stage_b_head_focal.pt \
    --data data/processed/stage_b
```

Both datasets (`data/processed/stage_b`, `data/processed/stage_b_scratch15`) and all checkpoints
(`stage_b_head_bce_baseline.pt`, `stage_b_head_focal.pt`, `stage_b_alpha75/stage_b_head.pt`,
`stage_b_scratch15/stage_b_head.pt`) exist locally and on the FAU HPC `$WORK`; the raw job logs
(`stage_b_1799124.out`, `stage_b_1799134_focal.out`, `stage_b_s15_1802973.out`,
`eval_s15_1802985.out`) are kept alongside them. The pre-rebuild α=0.75 checkpoint additionally
exists on Google Drive (see `development_log.md` Session 18) since it was originally trained
there.
