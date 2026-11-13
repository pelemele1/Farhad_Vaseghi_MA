---
title: SoilingNet — Soiling Detection on Automotive Surround-View Cameras
sources: [raw/papers/Uricar et al. - 2019 - SoilingNet Soiling Detection on Automotive Surround-View Cameras.pdf]
updated: 2026-08-31
---

# SoilingNet

Uřičář, Křížek, Sistu, Yogamani (Valeo R&D) — ITSC 2019, arXiv:1905.01492.
The paper Stage B's tile/grid design (architecture.md §2) is explicitly
modeled on. Full paper read.

## What it actually does

Formally defines automotive camera-soiling detection as **multi-label
classification** with two classes: **opaque** (region where nothing behind
the soiling is visible) and **transparent** (region that's blurred/distorted
but still shows something of the original scene through it). This is a
*different* taxonomy from this thesis's dirt/water/scratch — SoilingNet
classifies by how the soiling occludes light (opaque vs. see-through),
this project classifies by soiling *cause* (dirt/water/scratch). No direct
class correspondence; don't conflate the two schemes.

**The tile-level idea this thesis's Stage B actually borrows:** they
annotate soiling as coarse polygons, then generalize image-level
classification to **tile-level** by converting each polygon into a
per-tile label based on **percentage of tile area covered by the
polygon** — the same "rasterize mask → threshold per-tile coverage"
mechanism this project's own [[tile-grid-coverage-thresholding]] implements
in `src/soiling/tile_labels.py`'s `rasterize_tile_label`. They note tile
size 1×1 degenerates to pixel-level segmentation and tile size = full image
degenerates to image classification — the same generalization this
project's `grid_size = img_size // 32` design sits inside.

**Architecture:** a shared CNN encoder (ResNet10-like) feeding three
decoders in a multi-task network — object detection (simplified YOLOv2),
segmentation (simplified FCN8), and a new soiling decoder (2 conv layers +
grid-level softsign activation, tile sizes tested: 64×64 and full
resolution). Losses combined via weighted average, weights tuned by grid
search. This is architecturally different from Stage B (no multi-task
detection/segmentation heads here, just the single soiling decoder
concept, frozen-backbone strategy instead of joint multi-task training —
architecture.md's Option 2 is the multi-task path SoilingNet actually
uses, not the one this project picked for Stage B).

**Dataset:** 76,448 images (every 10th frame of video, to reduce
redundancy), manually annotated by polygon + opaque/transparent tag,
60/20/20 stratified split. Heavily imbalanced toward "clean." Front/rear
cameras get soiled far more than left/right (different mounting exposure).

**GAN-based augmentation:** used CycleGAN and MUNIT to synthesize
additional soiled images from clean ones (an alternative to this thesis's
physics-based synthetic approach — see [[procedural-lens-soiling]] for the
contrasting physics-based method this project actually uses). CycleGAN
converged and gave a modest accuracy gain (~3%); MUNIT did not converge to
usable results in their experiments.

## A directly relevant result: tile-level precision collapses relative to image-level

Table I in the paper reports tile-level (64×64) precision of only
**59–71%** across the four camera positions, despite recall near **99%**
in every case — the same "recall stays high, precision craters" pattern
this thesis's own Session 16 HPC baseline run hit for the scratch class
(precision 0.063, recall 0.852 — see `docs/development_log.md`). SoilingNet
doesn't diagnose *why* precision degrades at tile level the way this
project's own root-cause analysis did (pos_weight-driven over-prediction,
architecture.md's Stage B design note), but the paper's own numbers confirm
this isn't unique to this project's implementation — tile-level soiling
classification is intrinsically harder on precision than image-level
classification, at least with the class-imbalance handling both efforts
tried first.

## Open questions / not covered here

- The GAN augmentation section (CycleGAN/MUNIT training details) — read
  but not deeply ingested; would matter if this thesis ever explores GAN
  augmentation as an alternative/complement to the physics-based synthetic
  pipeline already in use.
- Table II/III's full confusion-matrix breakdown by camera position — read,
  not transcribed in full here (would be excessive detail for this page).
