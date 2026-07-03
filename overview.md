# Project Overview

**Goal:** Detect (and restore) image distortions — dirt, water, scratches on a protective glass in front of a camera — using a neural network. The project has two parts: creating a paired dataset, and training a multi-task network on it.

## Part 1 — Dataset Creation

Two rigidly mounted, hardware-synchronized cameras capture the same scene at 10–20 m. The right camera always looks through a glass pane (clean → reference, prepared → distorted); the left camera is the ground-truth reference. A single static homography (after per-camera lens undistortion) maps the right image into the left coordinate system, yielding almost pixel-aligned **(clean, distorted)** image pairs — no manual masks needed.
→ Details in [setup.md](setup.md).

## Part 2 — Network Training

A YOLO-based multi-task network with a **shared backbone + separate heads**: the primary head does object detection/segmentation, an added **distortion head** detects dirt/water/scratches. Start with an image-level distortion head (Stage A) trained on a frozen backbone (robust, no negative transfer); optionally scale to pixel-level segmentation (Stage C) using pseudo-masks derived for free from the difference of each (clean, distorted) pair.
→ Details in [architecture.md](architecture.md).

## Approach

Build a small pilot dataset first to validate the whole pipeline (mapping accuracy, distortion pairs, a quick model test), then scale to a larger, more diverse dataset. Prior art comes mainly from the automotive fisheye "lens soiling" literature (OmniDet, SoilingNet, WoodScape).
