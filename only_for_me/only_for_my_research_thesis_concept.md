# Thesis Concept: Distortion-Robust and Failure-Aware Perception from Paired Camera Data

Companion to [overview.md](overview.md), [setup.md](setup.md), [architecture.md](architecture.md),
and the prior-art analysis in [ablations.md](ablations.md).

---

## 1. Framing

**Research question:** Can a pixel-aligned (clean, distorted) image pair — produced by a
dual-camera rig — be used to train a single-camera network that is both (a) *robust* to
lens distortion (dirt, water, scratches on protective glass) and (b) *honest* about where
distortion makes its predictions untrustworthy?

**Why this is the right framing for an MA:** A clean-slate conceptual novelty is unlikely
in this mature field (see [ablations.md](ablations.md) — Desoiling/Valeo, Porav, OmniDet,
diffusion restoration, physics-based soiling simulation all exist). The defensible
contribution is *combinatorial but well-motivated*: a training signal that the existing
literature does not have, applied to a joint task.

**The central asset:** The pixel-aligned pair gives a supervisory signal nobody else has
cleanly:
- Desoiling/Valeo could **not** align their multi-camera images → fell back to unpaired CycleGAN.
- Porav is aligned but needs the reference **at test time** (restoration-as-preprocessing), rain only.
- Here the reference camera exists **only at training** and is removed for deployment.

---

## 2. Two Contributions (and their combination)

### C1 — Robustness via Privileged-Information Distillation
The clean reference is *privileged information* (Vapnik's LUPI): available at training,
absent at inference. A **teacher** sees the clean image; a **student** (the deployed model)
sees only the distorted image and is trained to reproduce the teacher's features. The
student learns distortion-invariant representations; the reference camera is "distilled away."

Distinct from [KD-for-restoration (2025)](https://www.researchgate.net/publication/390536799_Knowledge_Distillation_for_Image_Restoration_Simultaneous_Learning_from_Degraded_and_Clean_Images),
which distills clean→student for restoration only and is motivated by model compression —
here it is motivated by the physical two-camera privilege and applied to the *joint*
detection + distortion + restoration task on real aligned pairs.

### C2 — Failure-Aware Gating
Instead of emitting soiling as just another class, a **reliability head** predicts *where the
primary task will fail*. Its ground-truth target is generated automatically from the pair:
run the detector on clean vs. distorted, and the per-region degradation (lost boxes,
confidence/IoU drop) marks where distortion actually caused a failure. At inference the
reliability map gates the detector (raise uncertainty / flag "cannot see" / trigger cleaning),
converting silent false negatives into flagged unreliable regions.

Distinct from existing introspection / false-negative prediction work: the failure target is
**caused specifically by physical lens distortion** and supervised from the aligned clean
counterpart — a signal unique to this setup.

### C1 + C2 combined
Complementary: C1 makes the detector *robust* (fix what can be fixed); C2 makes it *honest*
(flag the residual). Same anchor — the aligned pair — used two ways. The combination is the
proposed thesis direction; C1 or C2 alone is a valid fallback scope.

---

## 3. Architecture (single shared backbone)

```
TRAINING
  Clean ref  ─► Teacher backbone T ─► F_T ─► teacher detection head
                                        │  distillation ‖F_S − F_T‖ (aligned, pixel-wise)
  Distorted  ─► Student backbone S ─► F_S ─┬─► detection head        (primary task)
                                           ├─► restoration decoder   (GT = clean image)
                                           └─► reliability head       (GT = clean-vs-distorted
                                                                        detector degradation)
INFERENCE (single camera)
  Distorted ─► Student S ─► detections + restoration + reliability map ─► gated output
```

Base: YOLO-style shared backbone (see [architecture.md](architecture.md)); teacher can be a
frozen copy pretrained on clean images. Restoration decoder optionally a transformer
(Restormer/Uformer-class) — but on pretrained weights only, given the small dataset.

---

## 4. Training Signals (all derived, no manual masks)

| Head | Ground-truth source |
|---|---|
| Detection | Standard labels on the primary detection dataset |
| Restoration | The clean reference image of the aligned pair |
| Distortion / reliability | (i) difference image (dirt/water/scratch present + coarse localization); (ii) detector degradation clean-vs-distorted (failure regions) |
| Distillation | Teacher feature maps on the clean image |

Distortion *type* label comes from the capture log (which prepared pane was mounted) — see [setup.md](setup.md).

---

## 5. Baselines to Compare Against

1. **Detection-only** on distorted images (no distortion handling) — lower bound.
2. **Data augmentation** — train the detector on synthetically soiled images (physics-based
   sim / GAN, cf. SoilingNet's "Let's Get Dirty"). The standard cheap alternative to beat.
3. **Restoration-as-preprocessing** (Porav-style) — restore first, then detect. The main
   competing paradigm; needs the reference or a restoration net.
4. **OmniDet-style frozen-encoder multi-task** — soiling as a separate bolted-on head.
5. **Ablations of the proposal** — C1 only, C2 only, C1+C2; with/without distillation;
   with/without gating.

---

## 6. Evaluation Metrics

- **Primary-task robustness:** detection mAP on distorted images (proposal vs. baselines).
- **Safety-relevant (the key metric):** false-negative rate in soiled regions **with vs.
  without** the reliability gate — does C2 turn silent misses into flags?
- **Restoration quality:** PSNR/SSIM vs. the clean reference (proposal vs. Porav-style).
- **Distortion detection:** per-region precision/recall (and IoU if pseudo-masks are used).
- **Deployment realism:** single-camera inference latency (rules out over-heavy transformers
  if real-time is a target — the original reason YOLO was chosen).
- **Generalization:** train on own data, evaluate on WoodScape soiling (and vice versa) to
  support any SOTA-adjacent claim across the domain gap.

---

## 7. Milestones (phased, de-risked)

| Phase | Goal | Exit criterion |
|---|---|---|
| M0 | Rig + sync + capture protocol | Synchronized clean/distorted pairs captured ([setup.md](setup.md)) |
| M1 | Mapping + validation | Reprojection/photometric residual below threshold; alignment good enough for feature distillation |
| M2 | Pilot dataset (~15–30 scenes × 2–3 types × 2 severities) | Pipeline validated end-to-end; pseudo-labels sane |
| M3 | Baselines reproduced | Detection-only + augmentation + OmniDet-style running on own data |
| M4 | C1 (privileged distillation) | Student beats detection-only under distortion; ablation shows distillation helps |
| M5 | C2 (failure-aware gating) | Gate reduces false negatives in soiled regions at acceptable trade-off |
| M6 | Combination + scale-up + WoodScape cross-eval | Full system + generalization evidence |

C1 or C2 alone (M4 **or** M5) is a complete MA scope; M6 is the stretch goal.

---

## 8. Risks & Mitigations

- **Alignment residual corrupts feature distillation** → gate M1 strictly; fall back to
  output-level KD (less alignment-sensitive) if feature-level is too noisy.
- **Dataset too small for transformers** → use pretrained restoration/backbone weights,
  fine-tune only; do not train from scratch.
- **Latency** vs. real-time target → keep detector YOLO-based; treat heavy restoration as
  optional/offline branch.
- **Teacher is imperfect GT** → teacher detections are a soft target, combined with real labels.
- **Novelty challenged in review** → position explicitly as combinatorial, delimited against
  Porav/Desoiling/OmniDet ([ablations.md](ablations.md)); lead with the unique training signal.

---

## 9. Scope / Non-Goals

- Not aiming for clean-slate conceptual novelty — see [ablations.md](ablations.md).
- Not aiming to beat automotive fisheye SOTA on their benchmark head-to-head (domain
  mismatch); WoodScape is used for generalization evidence, not as the primary benchmark.
- Fixed rig, negligible-parallax regime (10–20 m); depth-dependent mapping is out of scope.
