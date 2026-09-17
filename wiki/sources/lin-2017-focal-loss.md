---
title: Focal Loss for Dense Object Detection
sources: [raw/papers/Lin et al. - 2017 - Focal Loss for Dense Object Detection.pdf]
updated: 2026-08-31
---

# Focal Loss for Dense Object Detection

Lin, Goyal, Girshick, He, Dollár (Facebook AI Research) — ICCV 2017,
arXiv:1708.02002. Introduces the loss `FocalLossWithLogits` in
`src/models/losses.py` is a direct implementation of. Full paper read.
See [[focal-loss]] for the technique itself (this page is the paper
summary/citation; that page is the concept, generalized beyond this one
paper).

## The problem it addresses

One-stage dense object detectors (YOLO, SSD-style — evaluate ~100k
candidate locations per image) suffer extreme foreground:background class
imbalance, often ~1:1000. Two-stage detectors (R-CNN family) avoid this via
a proposal stage that discards most background before classification, plus
biased minibatch sampling. One-stage detectors can't cheaply do that, so
training gets dominated by a huge number of "easy negatives" — individually
tiny losses that sum to overwhelm the gradient and degrade the model.

The standard fix — a scalar α-balanced cross-entropy (weight the rare class
more) — only rebalances by *frequency*. It doesn't distinguish *easy* from
*hard* examples within a class, so it can't stop a flood of confidently-
correct easy negatives from still dominating training even after
reweighting.

## The loss itself

Focal loss adds a modulating factor `(1 - p_t)^γ` to cross-entropy:

`FL(p_t) = -(1 - p_t)^γ log(p_t)`

γ=0 reduces to plain CE. As γ increases, well-classified ("easy") examples
get progressively down-weighted, so training concentrates on hard,
misclassified examples. In practice they use the α-balanced variant:

`FL(p_t) = -α_t (1 - p_t)^γ log(p_t)`

— **this is exactly the formula `FocalLossWithLogits.forward` implements**
in `src/models/losses.py` (`alpha_t * (1 - p_t)**gamma * bce`). The paper's
own best setting, found by sweeping γ and α together on COCO/RetinaNet, is
**γ=2, α=0.25** — the exact default values `FocalLossWithLogits.__init__`
and `build_stage_b_focal_loss` use. Confirmed: this project's defaults
match the paper's reported best setting, not an arbitrary choice.

## A detail used in the paper but not yet in this project's code

§3.3 ("Class Imbalance and Model Initialization"): with extreme class
imbalance, initializing the final classification layer normally (~50/50
predicted probability) makes the frequent class's loss destabilize early
training. Their fix is a **bias initialization** on the last conv layer so
the *initial* predicted probability for the rare class is deliberately low
(they use π=0.01) — not a loss-function change, an initialization change.
They found this necessary for training to converge at all with cross-entropy,
and to modestly help focal loss too.

**`StageBDistortionHead`** (`src/models/distortion_head.py`) is currently
a plain `nn.Conv2d` with PyTorch's default initialization — this bias trick
is not applied. Worth flagging as an untried idea for the scratch class
specifically (its positive rate is the most extreme of the three classes,
~1%), separate from the focal-loss switch already made in Session 17.

## RetinaNet itself — not directly relevant here

The paper's proposed detector (ResNet+FPN backbone, twin classification/box
subnets) is a full object-detection architecture, not something this
project adopts — `FrozenYOLOBackbone` is a different backbone family
entirely. Only the loss function transfers to this project, not RetinaNet's
architecture.

## Results (context, not directly reused)

FL beat both α-balanced CE (2.9 AP gain) and online hard-example mining
(OHEM, the previous best technique for this problem, 3.2 AP gain) on COCO
object detection with RetinaNet. Not directly comparable to this project's
own tile-classification metrics, but establishes the technique's general
effectiveness for the class-imbalance problem it addresses.
