---
title: "YOLOv11: An Overview of the Key Architectural Enhancements"
sources: [raw/papers/Khanam and Hussain - 2024 - YOLOv11 An Overview of the Key Architectural Enhancements.pdf]
updated: 2026-08-31
---

# YOLOv11 architectural overview

Rahima Khanam, Muhammad Hussain (University of Huddersfield) —
arXiv:2410.17725 (2024). Full paper read. **Not an Ultralytics-authored
paper** — Ultralytics never published an official YOLO11 paper (unlike
v8/v9/v10, each of which has one); this is an independent third-party
architectural analysis. `weights/yolo11m.pt` (the weights
`FrozenYOLOBackbone` loads) is the model this paper describes.

## What actually changed vs. YOLOv8 (the backbone's lineage)

- **C3k2 block** replaces YOLOv8's C2f block throughout backbone, neck, and
  head. Uses two smaller convolutions instead of one large one (cheaper),
  with a `c3k` flag: `False` → behaves like C2f (standard bottleneck);
  `True` → swaps in the deeper C3 module for more complex feature
  extraction.
- **C2PSA block** (Cross Stage Partial with Spatial Attention) — new,
  added after the existing SPPF block. Adds a spatial-attention mechanism
  YOLOv8 didn't have, intended to help the model focus on relevant regions,
  particularly small or partially occluded objects.
- Backbone/neck/head three-part structure (feature extraction → multi-scale
  aggregation → prediction) is unchanged from the general YOLO lineage —
  YOLO11 is an efficiency/attention refinement of YOLOv8's architecture,
  not a structural redesign.
- Headline efficiency claim: YOLOv11m matches or beats YOLOv8m's COCO mAP
  while using **22% fewer parameters**.

## Relevance to this thesis

`FrozenYOLOBackbone` (`src/models/backbone.py`) loads `yolo11m.pt` and
uses it purely as a **frozen multi-scale feature extractor** — this
thesis's Stage A/B/C heads never touch YOLO11's own detection head, C2PSA
attention, or any of its task-specific outputs (bounding boxes, instance
masks, pose, OBB — §5 of this paper). Only the backbone's P5 feature map
(architecture.md's own design choice) is consumed. So the specific
architectural deltas in this paper (C3k2 vs C2f, C2PSA) matter to this
thesis only insofar as they affect the *quality of the frozen features*
the P5 map provides — not as design choices this project makes or could
tune, since the backbone is used pretrained and frozen, never modified or
retrained.

## Not covered in depth here

- §5's full breakdown of YOLO11's task variants (segmentation, pose,
  OBB, tracking) — irrelevant here since only the base backbone is used.
- §6/§7's benchmark comparison graphs (Figure 2, COCO mAP vs. latency
  across YOLOv5–v11) — read, not transcribed; would matter only if this
  thesis ever needed to justify the specific `yolo11m` size choice against
  smaller/larger variants, which hasn't come up.
