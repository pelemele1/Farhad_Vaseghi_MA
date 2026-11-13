---
title: Focal loss (technique)
sources: [raw/papers/Lin et al. - 2017 - Focal Loss for Dense Object Detection.pdf]
updated: 2026-08-31
---

# Focal loss

A loss-reweighting technique for extreme class imbalance: multiply
cross-entropy by a modulating factor `(1 - p_t)^γ` that shrinks as the
model's confidence in the correct class grows, so well-classified ("easy")
examples contribute progressively less to the loss and training
concentrates on hard/misclassified ones. For the paper it comes from and
the exact formula derivation, see [[lin-2017-focal-loss]] (that page is
the paper citation; this page is the generalized technique).

## Why it exists (vs. a flat class weight)

A flat positive-class weight (`pos_weight` in `BCEWithLogitsLoss`, or a
scalar α) rebalances loss by *frequency* only — it doesn't distinguish an
*easy*, already-confidently-correct example from a *hard*, borderline one
within the same class. With severe enough imbalance, a flat weight can
overcorrect: it makes the model chase the rare class's frequency-adjusted
loss everywhere, including on easy negatives, which can look like
over-prediction rather than accurate localization.

## Where this project used it

**Stage B's scratch class** (Session 16→17, `docs/development_log.md`):
the baseline `bce`+`pos_weight` run gave scratch a `pos_weight≈95` (its
tile-positive rate is only ~1%), and the trained head over-predicted
scratch almost everywhere — precision 0.063 despite recall 0.852. Switched
to `FocalLossWithLogits` (`src/models/losses.py`, γ=2, α=0.25 — the
paper's own best-found values) for the HPC run in job 1799134.

## An adjacent idea from the same paper, not yet applied

The paper's §3.3 bias-initialization trick (start the rare class's
predicted probability deliberately low, e.g. 0.01, via the final layer's
bias term, rather than default ~50/50 init) is a *different* lever than
focal loss itself — it changes model initialization, not the loss
function — and isn't implemented in `StageBDistortionHead` yet. Worth
trying if focal loss alone doesn't fully resolve the scratch class, since
it specifically targets the same "extreme imbalance destabilizes early
training" problem from a different angle. See [[lin-2017-focal-loss]] for
the detail.

## Relative to this thesis's earlier `pos_weight` approach

Both `pos_weight` (Stage A and Stage B's `bce` path,
`compute_pos_weight`/`compute_tile_pos_weight` in `src/models/losses.py`)
and focal loss address class imbalance, but at different granularity:
`pos_weight` reweights by the class's overall frequency (one number per
class), focal loss reweights *per example* by how confidently-correct the
model already is on it, regardless of class. They're not mutually
exclusive — the α-balanced focal loss variant this project uses still has
a per-class weight term (α) *on top of* the per-example (1-p_t)^γ term.
