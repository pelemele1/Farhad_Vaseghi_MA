---
title: "SIDL: A Real-World Dataset for Restoring Smartphone Images with Dirty Lenses"
sources: [raw/papers/12883-ChoiS.pdf]
updated: 2026-08-31
---

# SIDL

Sooyoung Choi, Sungyong Park, Heewon Kim (Soongsil University) — AAAI-25
(2025). Full paper read.

## What it is

A **real-world**, paired (degraded + clean) dataset for training/evaluating
*image restoration* (removing the effect of dirty lenses), not detection or
classification. 300 static scenes × 5 contaminant types (fingerprint,
dust, water drop, scratch, mixed) + a clean control condition, giving 1,588
degraded images total, each with a spatially-registered clean reference.

**Capture method — directly comparable to this thesis's rig concept**: a
thin PVC film with a refractive index/reflectance similar to glass is
physically contaminated (mud/water/scratch/etc. applied directly to the
film with real materials — a knife for scratches, a spray bottle for water,
hand cream for fingerprints, thread/sand between two films for dust), then
held close to the smartphone lens in a fixed 3D-printed film holder, with
the same static scene shot through clean and dirty film in turn. **This is
the same core idea as this thesis's own rig** (a distortable glass pane in
front of the lens, swapped between clean/dirty states) — SIDL validates
that approach's realism at smartphone scale, using a knife/spray-bottle/
hand-cream/thread-sand toolkit broadly consistent with
`docs/data_collection_pipeline.md`'s own distortion-pane recipes. A new
film was created per scene specifically to prevent restoration models from
memorizing fixed dirt patterns — the same anti-memorization concern this
project's own `add_dirt`/`add_water`/`add_scratch` per-variant randomization
already addresses via synthetic means instead of physical remaking.

## Task — different from this thesis's task

SIDL is a **restoration** benchmark (predict the clean image from the dirty
one, evaluated via PSNR/SSIM against the paired clean reference) — not
classification or localization. No direct code reuse for this thesis's
Stage A/B (multi-label / tile-grid classification), but it's the most
directly comparable *physical rig methodology* found in the papers ingested
so far — more comparable to Part 1 (this thesis's own capture rig) than to
Stage A/B's synthetic-training-data code.

## Findings that generalize beyond restoration

- Difficulty stratification by PSNR (`Easy`/`Medium`/`Hard`) against
  degradation type is a reusable idea for grading synthetic severity
  levels, conceptually similar to this project's own "light"/"heavy"
  severity split for synthetic dirt/water/scratch.
- Every restoration model tested (CNN, transformer, diffusion, Mamba-based)
  struggled significantly more on `Scratch` and `Fingerprint` than on
  `Dust`/`Water` at the `Hard` difficulty level — scratch consistently
  among the harder categories across architectures, a pattern with the
  same flavor (though a different task) as this thesis's own scratch-class
  difficulty in Stage B (Session 16).
- A **model trained on a narrower prior dataset (scratch-only) performed
  much worse on SIDL's own test set than a model trained on SIDL directly**
  — evidence that a synthetic/narrow-scope dataset doesn't transfer as well
  as a broader, more realistic one; a relevant caution for this thesis's
  reliance on synthetic MIO-TCD-based training data, though not directly
  tested here.

## Not covered in depth here

- Full quantitative benchmark tables (Table 2) across all 7 restoration
  models × 3 difficulty levels — read, not transcribed (would be excessive
  detail for a page whose relevance to this thesis is the *methodology*,
  not the restoration benchmark numbers themselves).
- The ISP/RAW-to-sRGB preprocessing pipeline details.
