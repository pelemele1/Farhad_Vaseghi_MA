# Prior Art & Novelty Assessment

Literature check on the two core ideas proposed for "beating SOTA":
(1) a real paired clean/distorted dataset via a dual-camera setup, and
(2) positive-transfer / feature-consistency for distortion-robust detection.

**Conclusion up front:** The core novelty is largely pre-empted. Each ingredient
already exists in published work. The realistic contribution is *incremental /
combinatorial*, not clean-slate — and must be explicitly delimited against the
works below.

---

## What already exists

### 1. Paired clean/soiled dataset via multiple cameras — essentially our approach
**Desoiling Dataset** (Uricar et al., ICCVW 2019, Valeo) —
[paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/AUTONUE/Uricar_Desoiling_Dataset_Restoring_Soiled_Areas_on_Automotive_Fisheye_Cameras_ICCVW_2019_paper.pdf)
- **4 cameras in a row, one kept clean, three manually soiled** → paired clean/soiled captures of the same scene.
- Targets restoration + soiling detection.
- **Key limitation:** they could **not** achieve pixel-level alignment (camera position shift → "non-affine perturbations"), so they fell back to **CycleGAN (unpaired)** instead of supervised paired learning.

### 2. Paired AND aligned, with downstream proof — even closer
**"I Can See Clearly Now: Image Restoration via De-Raining"** (Porav et al., ICRA 2019) —
[ar5iv](https://ar5iv.labs.arxiv.org/html/1901.00893)
- **Dual-chamber in front of a stereo camera** (one side clear, one side sprayed with real water droplets), **~50,000 pairs**, undistorted/cropped/**aligned via homography**.
- Demonstrates exactly the "our" idea I had proposed: **de-raining as a preprocessing step improves downstream segmentation** — better than re-training the segmenter on rainy images.

### 3. Two-glass trick + feature consistency
- Two identical glass panes, one prepared, one clean: published in
  [Seeing through Unclear Glass (2025)](https://arxiv.org/pdf/2509.01033),
  [Stereo Waterdrop Removal (IROS 2021)](https://cqf.io/papers/Stereo_Waterdrop_Removal_IROS21.pdf).
- Feature-consistency loss for robust detection:
  [Mind the Backbone](https://www.researchgate.net/publication/369556781_Mind_the_Backbone_Minimizing_Backbone_Distortion_for_Robust_Object_Detection).

---

## Honest reassessment

Every single ingredient of the proposed "novelty" is already demonstrated. This
does not make the project worthless — but the contribution is incremental, and
needs explicit delimitation.

| Idea | Status | Remaining gap |
|---|---|---|
| Paired data via multi-camera | Desoiling (Valeo) | Their alignment failed → CycleGAN. **We could deliver true pixel-accurate alignment** (10–20 m → negligible parallax) where Valeo failed. Real differentiator. |
| Paired + aligned + downstream | Porav (rain only, single camera axis) | Only **rain**, only **restoration→segmentation** as preprocessing. **Not** multi-type (dirt+water+scratch), **not** YOLO detection, **not** a single joint multi-task network. |
| Joint detection + distortion + restoration in *one* network with a positive-transfer proof on real aligned multi-type data | **not found as a single work** | Most plausible remaining niche — but each component exists individually. |

---

## Implications

- **As a pure dataset novelty it does not stand** — Desoiling and Porav are too close.
- **Realistic publishable contribution:** the *specific combination* — pixel-accurately aligned (where Valeo failed) + **multi-type** distortion (dirt/water/scratches, not just rain) + **a single joint network** (detection + distortion + restoration) with a demonstrated **positive transfer**. Defensible, but must explicitly delimit against and evaluate against Porav and Desoiling.
- **If the goal is application-oriented rather than paper-SOTA:** "it already exists" is actually good news — build on top of Porav / Desoiling / OmniDet instead of starting from scratch.

---

## Open follow-ups

- Delimit precisely against Porav & Desoiling (design table + evaluation protocol).
- Targeted search for whether the *full combination* (aligned + multi-type + joint detection+restoration) already exists somewhere.
