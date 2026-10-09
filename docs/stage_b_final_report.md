# Stage B Final Report — Tile/Grid Distortion Localization

> **Superseded design (Session 27).** The current pipeline — gate first (Stage A), one visible
> class per tile/pixel, color-coded maps and an image-level answer — is described in
> [`visible_pipeline_report.md`](visible_pipeline_report.md). This report remains the record of
> the multi-label models and of how the layers, heads and losses were chosen.

**Status:** complete. Canonical checkpoint: `checkpoints/stage_b_h32_lf/stage_b_head.pt`
(multi-scale tile head with 32 channels, focal loss α=0.75, γ=2.0, random left-right flips), trained on `data/processed/stage_b`
(14000 images, combo-inclusive, severity-balanced, severity-independent and audited ground
truth, §3), gated at inference time by `checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt`
at threshold **0.172** ([`stage_a_final_report.md`](stage_a_final_report.md) §5). The same dataset directory
also carries full-resolution pixel masks (`masks/`), which Stage C
([`stage_c_final_report.md`](stage_c_final_report.md)) trains against. For the full history
(loss-function iteration, threshold diagnostics, every intermediate checkpoint and dataset
rebuild, the Session 22 ground-truth fix, the Session 24–25 overfitting study, the Session 26 label audit), see [`development_log.md`](development_log.md); this
document reports only the current, final result.

---

## 1. Goal

Per `architecture.md` §2, Stage B extends Stage A's whole-image classification to **coarse
spatial localization**: "the feature map is treated as a grid (e.g. 16×16), one classification
output per tile" — so the model can say not just *whether* a distortion is present, but roughly
*where* on the lens, without needing pixel-mask annotation. Same frozen backbone and the same
three classes as Stage A; the output is one prediction per tile instead of one per image.

---

## 2. Pipeline overview

| Step | What | Where |
|---|---|---|
| 1 | Dataset builder (tile-grid ground truth, rasterized from each effect's own pixel mask) | `src/soiling/tile_labels.py`, `src/soiling/dataset_builder.py`, `scripts/build_stage_b_dataset.py` |
| 2 | Model: frozen multi-scale backbone + multi-scale tile head | `src/models/backbone.py`, `src/models/distortion_head.py::StageBMultiScaleHead` |
| 3 | Loss: focal (α=0.75, γ=2.0) | `src/models/losses.py` |
| 4 | Training (random left-right flips, keeps the best-validation epoch) | `scripts/train_stage_b.py --arch multiscale --hflip --hidden-dim 32` |
| 5 | FAU HPC (TinyGPU) submission | `scripts/hpc/stage_b_regularization.sh`, `stage_b_small_head.sh`, `run_shell.slurm` |
| 6 | Evaluation (per-tile precision/recall/F1/AP/ROC-AUC, tuned thresholds, gate, per-severity) | `scripts/evaluate_stage_b.py`, `scripts/diagnose_clean_false_positives.py`, `src/eval/` |
| 7 | Visualization (overlay grid + 8-column per-sample report) | `scripts/visualize_stage_b_results.py` |

---

## 3. Key details

### Tile ground truth

Each effect's own full-strength `[0,1]` pixel mask is area-average-pooled to the backbone's P5
grid — **16×16** at `--img-size 512` — and thresholded per tile: **dirt 0.20, water 0.25,
scratch 0.015** (scratch is a thin line and needs a much lower coverage cutoff than dirt/water's
filled regions). Stored as `tile_labels.npy`, `(N, 3, 16, 16)` uint8, row-aligned with
`metadata.csv`. A combo variant can have several classes positive in the same tile.

**Ground truth is severity-independent:** a tile is positive wherever the distortion *is*,
regardless of how faint it is in the image. Severity only changes the image (an alpha-blend
toward the clean photo), so a faint distortion must still be found — the project's explicit
goal. (An earlier version also scaled the mask by the severity alpha, which labeled most of a
faint patch as clean; see `development_log.md` Session 22.)

**Audited labels (Session 26).** Every image's labels were checked against the image itself
(`scripts/audit_dataset_labels.py`) and three defects fixed in place
(`scripts/fix_dataset_labels.py`); the generator now avoids all three:

- the thin-water mask of the vendored generator is `texture + 0.2`, so it marked the whole
  image as 20% water — 1966 of 6000 water masks, where 29% of the positive water tiles existed
  only because of that floor. The floor is removed from the mask (the rendering is unchanged);
- 9 variants had a correctly rendered distortion but an all-zero mask; they are re-rendered;
- 44 scratch labels named a scratch that is practically invisible (fewer than 20 pixels change
  by more than 10 gray levels — mostly outside the frame, or a bright highlight on white sky);
  those labels are removed.

Dirt lying under a later, strong water layer keeps its label (5.2% of dirt pixels lie under
high-severity water); it is physically on the lens, though often hard to see (§5).

### Model

`StageBMultiScaleHead`: the frozen YOLOv11-m backbone is tapped at four depths — stride 4, 8, 16
and 32 (P5). Each map is reduced to 32 channels by a 1×1 conv, area-pooled down to the 16×16
tile grid, concatenated, fused by a 3×3 conv (so each tile also sees its neighbors) and
projected to 3 per-tile logits — 95k trainable parameters. It is still
exactly one classification output per P5 tile; the difference from a head on P5 alone is that
each tile also sees the finer layers, where faint texture changes are still present. P5 alone
is deep and semantic, and loses most of that signal — the P5-only head reached only 0.20–0.27 AP
on low-severity distortions (§4).

### Layer study and scratch weighting

The design studies below (layers, overfitting, head size) were run before the label audit;
their conclusions are relative comparisons and unaffected, the absolute numbers are not
re-measured.

Identical training (focal α=0.75, 40 epochs, best-validation epoch kept, 64-channel head, no
augmentation) with different sets of backbone layers, compared on the **validation** split (AP dirt / water / scratch):

| layers (strides) | AP | mean AP | faint (low-severity) AP | mean faint AP |
|---|---|---|---|---|
| 32 only, same multi-scale head (control) | 0.805 / 0.780 / 0.491 | 0.692 | 0.506 / 0.431 / 0.314 | 0.417 |
| 16 + 32 | 0.898 / 0.888 / 0.733 | 0.840 | 0.717 / 0.667 / 0.577 | 0.654 |
| 8 + 16 + 32 | 0.922 / 0.914 / 0.793 | 0.876 | 0.779 / 0.737 / 0.647 | 0.721 |
| **4 + 8 + 16 + 32 (used)** | **0.932 / 0.921 / 0.822** | **0.892** | **0.806 / 0.775 / 0.691** | **0.757** |
| 2 + 4 + 8 + 16 + 32 | 0.933 / 0.924 / 0.810 | 0.889 | 0.808 / 0.779 / 0.663 | 0.750 |

The control row runs P5 alone through the same bigger head: it beats the original 1×1-conv
head but stays far behind, so the gain comes mainly from the finer layers, not from the head.
Localizing needs finer detail than whole-image classification (Stage A's best set stops at
stride 8); the stride-2 layer adds nothing here, since everything is pooled to 32-px tiles.

Weighting scratch more heavily in the loss (`--class-weights 1,1,2` / `1,1,3`) did not help:
scratch AP 0.820 / 0.825 (unweighted 0.822), while faint-water AP fell to 0.759 / 0.714
(unweighted 0.775). Not adopted.

### Overfitting and augmentation

Without augmentation the head overfits mildly: validation loss bottoms out at epoch 23 of 40
(0.0160) and rises to 0.0173 by epoch 40 while training loss keeps falling. The cause is the
small number of distinct scenes: 11200 training images, but only 800 different source photos
(14 variants each), so the head starts memorizing backgrounds. Keeping the best-validation
epoch already protects the saved model, but it stops at an only moderately trained head.

Each fix was added to the setup above and compared on the **validation** split:

| run | AP | mean AP | faint AP | mean faint AP | val loss: best → epoch 40 |
|---|---|---|---|---|---|
| no regularization (previous) | 0.932 / 0.921 / 0.822 | 0.892 | 0.806 / 0.775 / 0.691 | 0.757 | 0.0160 → 0.0173 |
| weight decay 0.05 (AdamW) | 0.932 / 0.921 / 0.830 | 0.894 | 0.803 / 0.775 / 0.703 | 0.760 | 0.0156 → 0.0172 |
| dropout 0.2 | 0.932 / 0.924 / 0.823 | 0.893 | 0.804 / 0.776 / 0.685 | 0.755 | 0.0158 → 0.0170 |
| **left-right flip** | **0.934 / 0.924 / 0.837** | **0.898** | **0.808 / 0.782 / 0.716** | **0.769** | **0.0151 → 0.0158** |
| flip + weight decay | 0.933 / 0.923 / 0.841 | 0.899 | 0.806 / 0.779 / 0.722 | 0.769 | 0.0150 → 0.0160 |
| flip + weight decay + dropout | 0.931 / 0.921 / 0.840 | 0.897 | 0.803 / 0.770 / 0.718 | 0.764 | 0.0153 → 0.0160 |

Weight decay and dropout only delay the overfitting. A random left-right flip (image and tile
labels mirrored together) attacks its cause by showing each scene in two versions, and gives
the lowest validation loss and the best faint-distortion AP. Adding weight decay changes nothing
beyond noise, and dropout costs faint accuracy, so the flip is kept. Other augmentations
are deliberately not used: brightness/contrast/color changes would alter what a faint
distortion looks like (the label would no longer match), crops or rescaling would break the
fixed tile grid, and a vertical flip is not a realistic camera view.

The flip stops validation loss from rising, but it still stalls around epoch 10 while
training loss keeps falling, so the train/val gap keeps widening. To close it, the head's
capacity was reduced (all runs with the flip; gap = mean train vs. val loss over the last 10
epochs):

| head | parameters | AP | mean AP | mean faint AP | train / val loss (gap) | best epoch |
|---|---|---|---|---|---|---|
| 64 channels, 3×3 fuse | 263k | 0.934 / 0.924 / 0.837 | 0.898 | 0.769 | 0.0120 / 0.0157 (0.0037) | 23 |
| **32 channels, 3×3 fuse (canonical)** | **95k** | **0.932 / 0.922 / 0.826** | **0.893** | **0.763** | **0.0139 / 0.0163 (0.0024)** | **38** |
| 16 channels, 3×3 fuse | 38k | 0.924 / 0.911 / 0.781 | 0.872 | 0.717 | 0.0159 / 0.0172 (0.0013) | 33 |
| 32 channels, 1×1 fuse | 62k | 0.897 / 0.876 / 0.782 | 0.852 | 0.641 | 0.0189 / 0.0200 (0.0011) | 37 |
| 16 channels, 1×1 fuse | 30k | 0.887 / 0.863 / 0.761 | 0.837 | 0.607 | 0.0205 / 0.0207 (0.0001) | 39 |
| 8 channels, 1×1 fuse | 15k | 0.875 / 0.847 / 0.726 | 0.816 | 0.566 | 0.0223 / 0.0218 (none) | 38 |

The gap only vanishes for heads too small to learn the task well (1×1 fuse: −0.05 to −0.08
mean AP, −0.13 to −0.20 faint AP). The 32-channel head is the chosen trade-off: 0.005 lower
AP, its validation loss keeps improving until epoch 38 instead of stalling, and the gap is about a
third smaller. On the test split it also raises far fewer false alarms on clean images (§4). A small
gap remains; closing it fully without losing accuracy would need more distinct source scenes.

### Loss and decision threshold

**Focal loss (α=0.75, γ=2.0).** Tile-level class imbalance is severe (scratch tiles ~2%
positive); plain `BCEWithLogitsLoss(pos_weight=...)` over-predicted scratch almost everywhere,
and α=0.75 beat both RetinaNet's default α=0.25 and a sum-of-squared-differences alternative.
Each class gets its own decision threshold, tuned on the val split for best F1
(`src/eval/thresholds.py`), since the three classes' precision/recall curves differ a lot.

### Inference-time gate

The image-level impaired gate ([`stage_a_final_report.md`](stage_a_final_report.md) §5) zeroes
every tile of an image it calls "not impaired" (`src/eval/gate.py::apply_gate`). Its threshold
(**0.172**) is chosen on the val split to keep ≥95% of impaired images
(`evaluate_impaired_gate.py --recall-target 0.95`) — a gate's job is to drop clean images
without discarding real distortions, so best-F1 is the wrong criterion for it. A gate false
negative still silently suppresses a genuine detection; its measured cost is in §4.

---

## 4. Results

HPC job `1834983` (V100), 40 epochs with random left-right flips on the audited labels, best
validation loss at epoch 38 (0.0150). Validation loss is noisy from epoch to epoch (±0.001) but
does not trend upward.

![Stage B training curve](images/stage_b_training_curve_h32.jpg)

**Per-tile metrics (held-out test split, 1400 images, 358400 tiles, tuned thresholds)**

![Stage B per-tile metrics](images/stage_b_test_metrics_h32.jpg)

| class | threshold | precision | recall | F1 | AP | ROC-AUC | support | gated F1 | gated AP | gated ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| dirt | 0.595 | 0.887 | 0.807 | 0.845 | 0.931 | 0.974 | 72748 | 0.844 | 0.928 | 0.970 |
| water | 0.593 | 0.813 | 0.849 | 0.831 | 0.917 | 0.972 | 72540 | 0.831 | 0.917 | 0.971 |
| scratch | 0.551 | 0.841 | 0.715 | 0.773 | 0.825 | 0.979 | 9015 | 0.751 | 0.774 | 0.901 |

**Effect of the label audit.** The same architecture trained on the old labels scores AP
0.933 / 0.921 / 0.829 on the old test labels, but 0.933 / **0.899** / 0.829 on the audited
ones: the floor-inflated water labels had overstated its water accuracy by about 0.02. Trained
on the audited labels, water recovers to 0.917; dirt and scratch move by less than 0.005 (within
run-to-run noise).

![Stage B ROC and precision-recall curves](images/stage_b_roc_pr_curves_h32.jpg)

**Clean-image false positives** (`diagnose_clean_false_positives.py`: share of the 103 clean
test images — 100 clean variants plus 3 whose only, invisible scratch label the audit removed —
with *any* of their 256 tiles above the class threshold, a strict measure):

| class | ungated | gated |
|---|---|---|
| dirt | 20% | **6%** |
| water | 21% | **11%** |
| scratch | 22% | **10%** |

The gate removes half to three quarters of these false alarms at almost no cost for dirt and
water; scratch pays the most (gated AP 0.825→0.774), because the gate occasionally misses a
thin scratch. Ungated, this model flags more clean images than the one trained on the old
labels (15% / 9% / 15% on the same images; 9% / 6% / 7% gated); with 103 images each percent
is a single image, and gated the two are close.

**Per-severity breakdown** (`--by-severity`, ungated, per tile) — how well are faint
distortions found?

| class | severity | precision | recall | F1 | AP | ROC-AUC | support |
|---|---|---|---|---|---|---|---|
| dirt | low | 0.830 | 0.617 | 0.708 | 0.804 | 0.955 | 25387 |
| dirt | medium | 0.833 | 0.878 | 0.855 | 0.938 | 0.988 | 23960 |
| dirt | high | 0.840 | 0.942 | 0.888 | 0.970 | 0.995 | 23401 |
| water | low | 0.681 | 0.674 | 0.678 | 0.735 | 0.953 | 23934 |
| water | medium | 0.743 | 0.913 | 0.819 | 0.922 | 0.989 | 23454 |
| water | high | 0.792 | 0.956 | 0.866 | 0.968 | 0.995 | 25152 |
| scratch | low | 0.828 | 0.536 | 0.651 | 0.663 | 0.961 | 2768 |
| scratch | medium | 0.817 | 0.746 | 0.780 | 0.831 | 0.988 | 3175 |
| scratch | high | 0.802 | 0.844 | 0.822 | 0.897 | 0.995 | 3072 |

Faint distortions remain the hardest, but the gap is now moderate: low-severity AP is 0.66–0.80
versus 0.89–0.97 at high severity. For comparison, the P5-only head scored against the same
labels reached only 0.27 / 0.24 / 0.20 low-severity AP (dirt / water / scratch) — the
multi-scale features roughly triple it. (On the audited labels the model trained on the old
labels reaches 0.808 / 0.739 / 0.666.)

![Stage B predicted probability by ground-truth severity (max per image)](images/stage_b_probability_by_severity_h32.jpg)
![Stage B sample predictions](images/stage_b_sample_predictions_h32.jpg)

A multi-distortion (combo) image, one panel per active class — several classes can be
positive in the same tiles:

![Stage B combo sample](images/stage_b_combo_sample_h32.jpg)

### Per-sample report figure

`scripts/visualize_stage_b_results.py::plot_stage_b_eight_column_report` — one row per sample:
original / distorted / ground-truth tiles / raw predicted probability (one column per class,
dominant one marked) / PR curve / gated probability. The raw columns and the PR curve always use
ungated predictions; the last column shows the post-gate result for direct comparison. A
multi-distortion image gets one row per class it contains, and its title names all of them.

![Stage B 8-column report, page 1](images/stage_b_full_report_h32_page1.jpg)
![Stage B 8-column report, page 2](images/stage_b_full_report_h32_page2.jpg)
![Stage B 8-column report, page 3](images/stage_b_full_report_h32_page3.jpg)

Clean images (page 1, rows 1–3) get low, noisy raw probabilities; the gate removes rows 2–3
and passes row 1. Dirt, scratches and water are localized closely (page 1 row 4; page 2). Page
1 rows 5–6 are one **dirt + water** image: the water on top is found, the medium dirt underneath
it is not — when two distortions overlap, the model reports the visible one (§5). Page 2 row 4
is a faint scratch the model finds but the gate erases. Page 3 is a water + scratch night
image: the scratch is found, the faint water film barely.

### General/any-distortion tile channel

A derived (not trained) "is *any* class active here" channel (`src/eval/general_channel.py`):
logical OR of the ground truth, noisy-OR `1 - Π(1-p_c)` of the predictions, and
`general_channel_consistency` as a self-check. A 4th trained channel would only re-learn a
deterministic function of the 3 the model already predicts.

---

## 5. Known limitations

- **Faint scratches are the weakest case** — low-severity scratch recall 0.54 (AP 0.66).
- **Overlapping distortions.** Dirt under a strong water layer is labeled but mostly invisible
  (it keeps 13% of its visibility under high-severity thin water, 22% under thick water); the
  model reports the water and misses the dirt there. This affects about 3–4% of dirt pixels.
- **Clean-image false alarms remain** at 6–11% of clean images even after gating (strict
  any-tile measure), and the gate trades that against occasionally suppressing a real
  detection (scratch gated AP 0.825→0.774).
- **Very small scratches** that cover less than 1.5% of every tile they touch have no positive
  tile (2 images); they are labeled at image and pixel level only.
- **Few distinct scenes.** 800 training photos. The flip and the smaller head keep validation
  loss from rising, but a small train/val gap remains; only more real variety would close it
  without losing accuracy.
- **Synthetic distortions only**, untested against real soiled-lens photos; small pilot
  dataset (1000 source photos) — same caveats as
  [`stage_a_final_report.md`](stage_a_final_report.md) §6.

---

## 6. Reproducing this report

```bash
python scripts/build_stage_b_dataset.py --source data/raw/mio_tcd/images \
    --out data/processed/stage_b --variants 14 --include-combos --include-severity \
    --dirt-threshold 0.20 --water-threshold 0.25 --scratch-threshold 0.015 --save-pixel-masks
python scripts/audit_dataset_labels.py --stage-a data/processed/stage_a --stage-b data/processed/stage_b
python scripts/train_impaired_gate.py --data data/processed/stage_b --arch multiscale \
    --img-size 512 --epochs 20 --device cuda --out checkpoints/impaired_gate_multiscale_lf
python scripts/train_stage_b.py --data data/processed/stage_b --arch multiscale --hflip \
    --hidden-dim 32 --fuse-kernel 3 --epochs 40 --loss focal --focal-alpha 0.75 --focal-gamma 2.0 \
    --device cuda --out checkpoints/stage_b_h32_lf
python scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b_h32_lf/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds --by-severity \
    --gate-checkpoint checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt --gate-threshold 0.172
python scripts/diagnose_clean_false_positives.py --checkpoint checkpoints/stage_b_h32_lf/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds \
    --gate-checkpoint checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt --gate-threshold 0.172
python scripts/visualize_stage_b_results.py --checkpoint checkpoints/stage_b_h32_lf/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds \
    --gate-checkpoint checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt --gate-threshold 0.172 \
    --log-file stage_b_lf_1834983.out --tag _h32 --out-dir docs/images
```

The current builder writes severity-independent, audited ground truth directly. The on-disk
dataset was built earlier and converted in place: `scripts/make_gt_severity_independent.py`
(Session 22), then `scripts/fix_dataset_labels.py` (Session 26; its changes are listed in the
dataset's `label_fixes.csv`). `build_b_severity_1822268.out` / `build_b_pixel_masks_1822558.out`
(dataset build), `stage_b_lf_1834983.out` (training) and `eval_lf_1834990.out` (test evaluation,
clean-image check and figures; `scripts/hpc/evaluate_after_label_fix.sh`) are the raw stdout of
the TinyGPU jobs; `eval_old_on_fixed_1835006.out` scores the pre-audit models on the audited
labels; the overfitting study is `scripts/hpc/stage_b_regularization.sh` and
`stage_b_small_head.sh` (logs `study_logs/b_reg_*`, `study_logs/b_small_*`). The pre-audit
model is kept as `checkpoints/stage_b_h32`, the earlier 64-channel models as
`checkpoints/stage_b_multiscale` (no flips) and `checkpoints/stage_b_flip`. Superseded checkpoints and their numbers are in
`docs/development_log.md`.
