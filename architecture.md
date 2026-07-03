# Concept: Multi-Task YOLO with Distortion Head

## 1. Overall Architecture

Design follows the OmniDet / FisheyeMultiNet principle: **one shared backbone + neck, three parallel heads.**

```
                    ┌─────────────────────────────────────┐
   Input (RGB) ───► │  Shared Backbone (YOLO, e.g. v11-m)  │
                    │  CSPDarknet / C2f stages             │
                    └───────────────┬─────────────────────┘
                                    │  feature maps P3/P4/P5
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
            │ Detection /  │ │ (opt.)      │ │ Distortion Head  │
            │ Segmentation │ │ Segment     │ │ (NEW)            │
            │ Head (YOLO)  │ │ Head        │ │                  │
            └──────────────┘ └─────────────┘ └──────────────────┘
              Primary task                     dirt / water / scratch
```

**Why a shared backbone:** The backbone learns generic features; both tasks share it → lower compute (single forward pass), and distortion detection can benefit from the semantic features. This is exactly the principle OmniDet demonstrated with a ~1% TPR loss at greatly increased system-level efficiency.

---

## 2. The Distortion Head — Three Expansion Stages

Recommendation: **start small** and only scale up when needed. All three use the same backbone.

### Stage A — Image-Level Classification (start, minimal effort)
- Global average pooling over the P5 feature map → small FC head.
- Outputs, two variants:
  - **Multi-label** (recommended): one sigmoid each for `dirt`, `water`, `scratch` → multiple distortions possible simultaneously.
  - Multi-class severity: `clean / mild / severe`.
- **Fits exactly the "image pairs only, no masks" constraint** — the label comes directly from your capture protocol (which glass pane was mounted in front of the camera).

### Stage B — Tile / Grid Classification (coarse localization, SoilingNet approach)
- The feature map is treated as a grid (e.g. 16×16), one classification output per tile.
- Provides coarse localization ("dirt in the top right") without pixel-mask annotation.

### Stage C — Pixel-Level Segmentation (OmniDet approach, highest precision)
- Small FCN/UNet-style decoder → per-pixel classes.
- **Trick for your setup:** You do **not** need to annotate the masks manually. From the mapped pair (distorted right image → left coordinate system) and the clean reference image, a pseudo-mask can be generated automatically via **difference image + threshold/morphology**. This is pixel-level ground truth "for free" — the actual advantage of your camera setup.

**Recommendation:** Start with **A** (validate the pipeline), then move to **C** via difference-image pseudo-labels — this is the best cost/benefit point for you, because the masks are essentially free.

---

## 3. Training Strategy — The Central Decision

Your distortion dataset is produced **separately** from the detection dataset (different scenes, different capture process). This is exactly the situation OmniDet solved. Two options:

### Option 1 — Frozen Backbone (recommended for the start, robust)
```
Phase 1:  Train backbone + detection/segment head on standard
          data (e.g. COCO-pretrained + your detection set)
              │
              ▼
Phase 2:  FREEZE the backbone, train only the distortion head
          on your image pairs
```
- **Advantage:** No negative transfer possible — the primary task is never degraded by distortion training. Fast, stable.
- **Disadvantage:** Backbone features are not optimized for distortion → possibly slightly lower distortion accuracy.
- OmniDet reports: asynchronously retraining the soiling decoder with a frozen encoder gave the same accuracy as more expensive joint training.

### Option 2 — Joint Multi-Task (max accuracy, more effort)
- All heads together, combined loss. Only sensible if you have images with **both** label types or work with multi-dataset sampling.
- Requires **loss weighting** (see below) and careful tuning against negative transfer.

**Recommendation:** Start with **Option 1**. Only switch to Option 2 (with cautious backbone fine-tuning at a small LR) if distortion accuracy is insufficient.

---

## 4. Loss Design

**Total loss (for joint training):**
```
L_total = w_det · L_detection  +  w_dist · L_distortion
```

**Head losses:**
- Detection/segmentation: standard YOLO loss (box + cls + dfl, or mask).
- Distortion Stage A: **BCE-with-logits** (multi-label). For class imbalance (lots of "clean") → `pos_weight` or focal loss.
- Distortion Stage C: Dice + BCE (standard for segmentation with small distortion regions).

**Do not naively set the weights `w_det`, `w_dist` to 1:1.** Established methods from the literature:
- **Uncertainty weighting** (Kendall & Gal 2018) — learns the weights as parameters, a good default.
- **VarNorm** (OmniDet) — weight ∝ inverse of the loss variance over the last ~5 epochs; the most robust in the OmniDet comparison.
- With a frozen backbone (Option 1) the problem disappears entirely — only one head is trained.

---

## 5. Data Flow / Label Provenance (your setup)

```
Right camera (distortion)  ──mapping (homography)──►  Left coordinate system
        │                                                      │
   distorted image                                    clean reference image
        │                                                      │
        └──────────────► difference image ◄────────────────────┘
                              │
                    threshold + morphology
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
         image label (A)          pseudo-mask (C)
      "dirt+water present"     per-pixel soiling GT
```

The distortion type (dirt/water/scratch) comes from your **capture log** (which prepared glass pane was mounted) — not from manual annotation.

---

## 6. Concrete Implementation Base

| Component | Recommendation | Rationale |
|---|---|---|
| Framework | **Ultralytics YOLOv11** (or v8) | Backbone/neck/head modular, custom head easy to add, segmentation variants ready |
| Backbone size | `-m` (medium) to start | Balance of accuracy/latency; scalable |
| Distortion head | Custom `nn.Module` on P5 (Stage A) | GAP + 2× FC + sigmoid, ~a few lines |
| Pretraining | COCO-pretrained backbone | Feature head start, then your data |
| Pseudo-masks | OpenCV: `absdiff` + Otsu + `morphologyEx` | From the existing camera pair, no annotation effort |

**Ultralytics implementation detail:** The cleanest way is to define an additional head entry in the model YAML that points to a backbone layer, plus a custom head class and an extended loss/dataloader that provides both label types. With **Option 1 (frozen)** it is even simpler: a separate small model that takes the frozen YOLO backbone features as input — no intervention in the YOLO training loop needed.

---

## References

- **OmniDet** (Kumar et al., RA-L + ICRA 2021, arXiv:2102.07448) — architecture and training strategy (shared encoder + synergized decoders, VarNorm loss weighting, frozen-encoder soiling training).
- **SoilingNet / FisheyeMultiNet** (Das et al., 2019, arXiv:1905.01492; ITSC 2019) — task formulation (ResNet-10 encoder + YOLO-V2 detection + FCN8 segmentation + soiling decoder).
- **WoodScape** (Yogamani et al., ICCV 2019) — public fisheye multi-task dataset with soiling classes (Clear / Transparent / Semi-Transparent / Opaque).
