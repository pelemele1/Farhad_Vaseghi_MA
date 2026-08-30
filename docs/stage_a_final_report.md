# Stage A Final Report — Image-Level Distortion Classification

**Status:** complete. **Date:** 2026-08-24.

This is a standalone summary of Part 2, Stage A — distinct from
[`development_log.md`](development_log.md)'s chronological session-by-session record. See
the log for the full narrative (decisions, dead ends, exact bugs and fixes); this document
is the result.

---

## 1. Goal

Per `architecture.md` §2, Stage A is the first and simplest of three planned distortion-head
designs: **image-level multi-label classification** of camera-lens soiling — `dirt`, `water`,
`scratch` — using global average pooling over a shared backbone's deepest feature map, a
small classification head, and a **frozen** (not fine-tuned) COCO-pretrained YOLO backbone
(architecture.md §3, "Option 1"). The purpose is explicitly to validate the whole pipeline
cheaply before any heavier investment (Stage B tile classification, Stage C pixel-level
segmentation, or unfreezing the backbone).

Concretely, this phase had to answer: **given a single traffic-camera image, can a
lightweight model reliably say whether the lens shows dirt, water, and/or a scratch?**

---

## 2. Pipeline overview

| Step | What | Where |
|---|---|---|
| 0 | Scratch-effect method decision | `src/soiling/effects.py` |
| 1 | MIO-TCD pilot subset (1000 images) | `src/data/mio_tcd.py`, `scripts/sample_mio_tcd.py` |
| 2 | Distortion synthesis (dirt/water vendored, scratch custom) | `src/soiling/effects.py`, `third_party/physical_lens_soiling/` |
| 3 | Stage A dataset builder (balanced, single-distortion variants) | `src/soiling/dataset_builder.py`, `scripts/build_stage_a_dataset.py` |
| 4 | Model: frozen backbone + distortion head + loss | `src/models/backbone.py`, `distortion_head.py`, `losses.py` |
| 5 | Training script | `scripts/train_stage_a.py` |
| 6 | FAU HPC (TinyGPU) setup/submission | `scripts/hpc/`, `docs/hpc_stage_a.md` |
| 7 | Evaluation | `scripts/evaluate_stage_a.py` |
| 8 | Test coverage audit | `tests/` (51 tests) |

---

## 3. Key details per section

### Distortion synthesis

- **`dirt`** and **`water`** (three mechanisms: thick stain, thin stain, droplet refraction)
  come directly from `physical_lens_soiling` (github.com/JannLi/physical_lens_soiling),
  vendored into `third_party/` with the supervisor's explicit permission — not
  reimplemented, only wired into a consistent interface.
- **`scratch`** has no equivalent in that repo. After comparing it against two external
  candidates (FilmDamageSimulator, ScratchSim — neither transferred cleanly to a flat lens
  scratch), a custom procedural generator was built instead: broken/discontinuous streaks
  (validated against a real photo of a scratched camera lens) with width that meanders
  between thin and thick along each streak, rather than a single uniform stroke.
- All three effects share one interface — `add_dirt(image, seed=None)`,
  `add_water(image, seed=None)`, `add_scratch(image, seed=None)` — each returning
  `(distorted_image, mask)`, with severity randomized internally rather than as call-time
  parameters.

### Dataset

- 1000 MIO-TCD traffic-camera frames (reproducibly sampled from the official
  `MIO-TCD-Localization` archive, seed=0), each producing **exactly 4 variants**: one clean
  + one each of dirt/water/scratch — **never combined on the same image** (an explicit
  design requirement). This guarantees an exact 25%/25%/25%/25% class balance by
  construction, rather than leaving it to chance.
- **4000 images total**, split **3200 / 400 / 400** (train/val/test) by *source photo*, not
  by variant — so no two variants of the same underlying scene can end up on opposite sides
  of the split. Verified directly: zero source-photo overlap between any two splits.

### Model

- **Backbone:** Ultralytics YOLOv11-m, COCO-pretrained, entirely frozen (`requires_grad =
  False` on every parameter, permanently kept in `eval()`). Traced its internal layer graph
  to confirm the true P5 tap: layer 10 (`C2PSA`) is the last purely-sequential layer before
  the FPN/PAN neck begins — 512 channels, stride 32.
- **Head:** GAP → FC(512→64) → ReLU → FC(64→3), returning raw logits (architecture.md §6:
  "GAP + 2× FC + sigmoid" — the sigmoid is applied by the loss function during training and
  explicitly at inference, not as a layer in the head, since `BCEWithLogitsLoss` needs
  logits for numerical stability).
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` computed directly from the training
  split's own class frequencies (came out to `[3.0, 3.0, 3.0]`, matching the exact 25%
  positive rate per class).
- Only the head's parameters are ever passed to the optimizer.

### Training

- Ran for real on the user's NHR@FAU TinyGPU allocation (account `iwnt196h`), one RTX3080
  GPU, 20 epochs, **12 minutes** wall-clock, 2.9 GB / 10 GB peak GPU memory.
- Train loss: **0.653 → 0.146** (smooth, monotonic). Val loss: **0.431 → 0.205** (noisy
  epoch-to-epoch but no runaway overfitting).
- Three real environment bugs were found and fixed only by actually running on the cluster
  (not reproducible locally) — a `--smoke-test` flag silently forcing CPU even with
  `--device cuda`, a missing `setuptools` pin (`pythonperlin` needs `pkg_resources`, which
  newer `setuptools` releases have removed entirely), and an `a100`-partition job that sat
  pending forever because this account's allocation has no GPU quota there. All three are
  documented in `development_log.md` Session 10.

---

## 4. Results

### Training curve (real TinyGPU run, job 1791674)

![Stage A train/val loss over 20 epochs on the RTX3080 TinyGPU run](images/stage_a_training_curve.jpg)

Train loss drops smoothly and monotonically throughout (0.653 → 0.146). Val loss drops
sharply for the first ~7 epochs (0.431 → 0.229), then plateaus around 0.20-0.24 with
epoch-to-epoch noise rather than continuing to fall or turning upward — the head has
essentially converged by epoch ~10-14, and the remaining epochs mostly fine-tune within that
band rather than overfitting. There's no held-out **test**-split curve during training by
design (standard practice: the test split is only touched once, for the final evaluation
below, not monitored during training where it could otherwise influence decisions).

### Per-class metrics (held-out test split, 400 images, 100 positive per class)

![Stage A per-class precision/recall/F1/AP on the held-out test split](images/stage_a_test_metrics.jpg)

| class | precision | recall | F1 | AP | support |
|---|---|---|---|---|---|
| dirt | 0.943 | 1.000 | 0.971 | 1.000 | 100 |
| water | 0.884 | 0.990 | 0.934 | 0.996 | 100 |
| scratch | 0.713 | 0.870 | 0.784 | 0.920 | 100 |

**dirt** and **water** are effectively solved by this frozen-backbone + tiny-head setup.
**scratch** is measurably weaker — plausible, since it's both the subtlest visual signal of
the three (thin lines vs. large textured regions) and the one custom-built effect rather
than the paper's validated code.

### Predictions on real test images

12 test-split images the model never saw during training, 3 per label (clean/dirt/water/
scratch), each captioned with its ground-truth label, the model's prediction at threshold
0.5, and the raw per-class probabilities:

![Real Stage A predictions on 12 held-out test images, 3 per class](images/stage_a_sample_predictions.jpg)

All 12 are classified correctly, most with high-confidence probabilities near 0.0 or 1.0.
(Regenerate with `python scripts/visualize_stage_a_results.py` — a different `--seed` picks
a different, equally representative sample.)

---

## 5. Known limitations

- **The backbone was never fine-tuned.** All learning happened in a ~33k-parameter head on
  top of frozen COCO features — the strong scores say those generic features already
  separate these distortion types well, not that the backbone understands lens soiling.
- **Trained only on single-distortion images.** No training example ever combined two
  distortion types on one image, by design. The model's sigmoid output *can* express
  multiple simultaneous positives, but that combination was never demonstrated during
  training — a real image with both dirt and water at once is untested territory.
- **Purely synthetic distortions.** `physical_lens_soiling`'s renders (and this project's
  own scratch generator) are a physics-inspired approximation, not real camera captures.
  There's an unmeasured domain gap between this and an actual soiled lens.
- **Small pilot dataset.** 1000 source photos, 4000 total training images — enough to
  validate the pipeline, well short of the scale a production model would use.

---

## 6. Reproducing this report

```bash
python scripts/evaluate_stage_a.py --checkpoint checkpoints/stage_a/stage_a_head.pt \
    --data data/processed/stage_a --split test
python scripts/visualize_stage_a_results.py --checkpoint checkpoints/stage_a/stage_a_head.pt \
    --data data/processed/stage_a --split test --log-file stage_a_1791674.out
```

`stage_a_1791674.out` is the raw stdout of the actual training job (`squeue.tinygpu` job ID
`1791674`), still sitting in `$WORK/Farhad_Vaseghi_MA/` on the cluster; `--log-file` is
optional and only needed to regenerate the training-curve plot.

Both scripts, the trained checkpoint, and the full dataset already exist locally and on the
FAU HPC `$WORK` — see `docs/hpc_stage_a.md` for cluster paths.
