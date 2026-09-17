---
title: Procedural generation of lens soiling data via physics-based simulation
sources: [raw/papers/Zhang et al. - 2025 - Procedural generation of lens soiling data via physics-based simulation.pdf]
updated: 2026-08-31
---

# Procedural generation of lens soiling data via physics-based simulation

Song Zhang, Rui Zhang, Jian Li, Xuefeng Li (Z-One Technology / Tongji
University) — *The Visual Computer* 41:10433–10449 (2025). **This is the
source paper for `third_party/physical_lens_soiling/`** — the vendored
code this thesis's `add_dirt`/`add_water` (`src/soiling/effects.py`) are
built directly on top of (see `third_party/physical_lens_soiling/NOTICE.md`).
Full paper read.

## The core idea

Decomposes the physical imaging process of a soiled lens into discrete
optical steps — light transmission through the soiling, scattering,
reflection off the soiling material, refraction — and approximates *each
step* with a basic OpenCV/scikit-image function, rather than using a full
physical renderer (Blender etc., accurate but slow/costly) or a GAN
(realistic but not physically interpretable, needs training data). Final
image = a weighted composition of the clean background (attenuated by a
`dirt_mask`) and a synthesized "dirt image" built from those approximated
optical terms, followed by a global brightness adjustment for the reduced
effective aperture.

Concretely: scattering ≈ Gaussian blur; reflection off the soiling ≈ a
Phong-style base-color/light-color composition; water-droplet refraction ≈
grid distortion via `skimage.transform.PiecewiseAffineTransform` (control
points on concentric polygons, magnified outward from image center) — the
exact function `add_droplet_distort.py`'s `add_distort` (vendored into
this project) calls. Mask shapes for dirt/water are generated with **Perlin
noise via the `pythonperlin` library**.

## Direct link to a known bug already documented in this project

This paper's own mask-generation choice — `pythonperlin` for procedural
noise shapes — **is the root cause of the pixel/label reproducibility gap**
this project's Session 6 documented: `pythonperlin` calls `np.random.seed(None)`
internally, silently reseeding the global RNG, which is why
`add_dirt`/`add_water` are only label-reproducible (same seed → same
boolean "soiled y/n") but not pixel-reproducible (same seed → different
exact mask) across separate calls. This wasn't a bug introduced by
adapting the code — it's inherent to the upstream method's dependency
choice, confirmed by reading the source paper rather than just the vendored
code. This is exactly why Stage B's dataset builder generates the mask and
rasterizes it into a tile label in the *same* function call that produces
the image (architecture note in the Stage B plan) — regenerating the mask
afterward from the "same" seed would not reproduce the same pixels.

## Effects generated (mapped to this project's taxonomy)

Paper generates: mud stains, water stains, water mist, water droplets
(refraction), lens flare, dust — a finer-grained taxonomy than this
project's three classes. `add_mud.py`'s `add_mudByTxture` (mud →
this project's `dirt`) and `add_dirtwaterByTxture`/`_slight` (water_thick /
water_thin → this project's `water`, blob-blend variant) map onto this;
`add_droplet_distort.py`'s `add_distort` is the water-droplet refraction
variant, a second, physically different mechanism also folded into this
project's single `water` class.

## Results relevant to this thesis's design choices

- **Classification task**: synthetic-only training reached 92.10% top-1
  accuracy vs. 93.70% for real data, and *exceeded* real-only when mixed
  (95.28%) — validates the physics-based approach as a reasonable
  real-data substitute/supplement for a classification task, which is what
  Stage A and Stage B both are.
- **Segmentation task**: synthetic-only training reached only ~58.81% mIoU
  vs. 93.11% for real data — a much larger sim-to-real gap for pixel-level
  ground truth than for classification. **Relevant caution for this
  thesis's future Stage C** (pixel segmentation, not yet started): the
  same underlying synthetic-mask generation this project already depends
  on showed a substantially bigger reality gap once evaluated at pixel
  granularity, in the source paper's own experiments.
- **Ablation**: removing the "ambient lighting" term (light passing through
  the soiling, not reflecting off it) hurt classification accuracy the
  most of any single term removed (92.10% → 71.22%) — the most
  load-bearing individual physical term in their model.
- **Speed** (single-threaded CPU): mud/water/flare effects are fast
  (0.13–0.16s/image); water-droplet refraction is slow (24.3s/image) —
  corroborates `third_party/physical_lens_soiling/UPSTREAM_README.md`'s own
  note that the droplet-distortion feature "runs slowly."

## Not covered here

- The InternImage benchmark model details (§4.1) — not relevant to this
  project's own backbone choice.
- Full ablation table (Table 6) beyond the one result cited above.
