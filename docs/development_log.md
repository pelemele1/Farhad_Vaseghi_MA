# Development Log

Chronological record of what was done and why, so progress can be followed without
digging through git history. One entry per development session/phase.

---

## Session 1 — Scope reset: Stage A on MIO-TCD, scratch-effect method choice

**Date:** 2026-08-23

`overview.md`, `architecture.md`, `setup.md`, and
`only_for_me/only_for_my_research_thesis_concept.md` / `..._ablations.md` are the source
of truth for what gets built (`README.md` is an unrelated generic template).

**Scope for this pass**, per architecture.md §2 and the supervisor's guidance:
- Dataset: **MIO-TCD** (Miovision Traffic Camera Dataset), not BDD100K.
- Target: Part 2, **Stage A only** — image-level multi-label distortion classification
  (`dirt` / `water` / `scratch`), GAP + FC + sigmoid on the YOLO backbone's P5 feature map,
  frozen COCO-pretrained backbone (no detection fine-tuning on MIO-TCD in this pass).
- Distortion source: synthetic, via https://github.com/JannLi/physical_lens_soiling for
  `dirt`/`water` — overview.md explicitly names this repo as a sanctioned alternative to
  the real dual-camera rig.
- Training destination: NHR@FAU HPC (TinyGPU cluster, SLURM). Per standing project rule,
  Claude prepares code and job scripts only — training itself runs on the user's own HPC
  allocation, never launched from here.

**Gap:** `physical_lens_soiling` has no `scratch` effect, and no ready-made lens-scratch
codebase exists. Resolved by building and comparing three candidates on one sample image
before committing to any of them for the full dataset build.

### Scratch-method comparison

1. **FilmDamageSimulator** (`github.com/daniela997/FilmDamageSimulator`, MIT licensed).
   Cloned and ran directly — composited 3 real scratch crops (extracted from their
   annotated 35mm film-scan dataset) plus 3 outputs of their own procedural
   `line_scratch()` (Perlin-noise-based fading line) onto the sample image. Needed one
   one-line fix for a NumPy 2.x incompatibility (`int()` on a size-1 array), otherwise ran
   as-is. Visual character: convincing as *film-grain* damage, but reads as photographic
   scratch texture rather than a lens/glass scratch.
2. **ScratchSim** (arXiv:2607.27065). Its public repo
   (`github.com/saptarshineil/ScratchSim`) turned out to be a full BlenderProc **3D**
   pipeline rendering a specific toy-car asset with material/normal-map perturbation —
   none of it applies to flat 2D photos. What was actually tested is a from-scratch
   reimplementation of just the paper's described 2D scratch-mask step (cubic Bézier
   curve through 4 random points + Gaussian blur), since that's the only part of their
   method that transfers to our setting.
3. **Own procedural generator** (`src/soiling/effects.py::add_scratch`) — built in the
   same mask + alpha-blend style `physical_lens_soiling` already uses for its other
   effects. Iterated twice against user feedback:
   - v1: smooth curve, tapered width, single continuous stroke → looked too clean/uniform.
   - v2: added random on/off gaps along the curve (a real lens scratch only catches light,
     and so is only visible, intermittently) plus bright "glint" points at a few spots —
     validated against a real reference photo of a scratched camera lens the user found
     (Lensrentals.com, "Front Element Scratches"), which visibly shows a discontinuous,
     not-solid scratch line.
   - v3 (final): removed the glint points (looked artificial/unnatural per user feedback)
     — kept the broken/discontinuous segments, and replaced the single global taper with a
     width that meanders between thick and thin along the streak (several blended random
     Gaussian "lobes"), which reads as natural irregular scratch depth without needing an
     explicit highlight marker.

**Result — v1 vs. v2 vs. v3** (same sample photo, all three methods; own generator in the
bottom row of each):

![v1: smooth tapered curve, single continuous stroke, too clean/uniform](images/scratch_comparison_v1_initial.jpg)

![v2: broken segments + glint points, rejected as artificial](images/scratch_comparison_v2_with_glints.jpg)

![v3 final: broken segments + meandering thick/thin width, no glints](images/scratch_comparison_v3_final.jpg)

**Decision: own procedural generator.** Rationale — full parametric control (count,
length, width, curvature, opacity all independently tunable for later class-balance
tuning), no external code/license/domain-mismatch dependency (FilmDamageSimulator = film
grain, ScratchSim = industrial 3D render, neither is a lens/glass scratch), composes
naturally with `physical_lens_soiling`'s own mask+blend convention for `dirt`/`water`, and
— unlike either external option — was directly iterated against real-world reference
evidence of what a scratched lens actually looks like.

**Follow-up tweak:** the v3 renders above all looked uniformly thin. Widths were already
random per scratch, but the default range (`width_px=(1, 3)`) was too narrow, and the
meandering-width profile only reaches its sampled max briefly near each "lobe" center — so
in practice almost every stroke stayed close to 1px regardless of seed. Widened the default
to `width_px=(1, 7)` so a clearly thick scratch is actually reachable. Confirmed across 6
random seeds that both thin and thick streaks now occur:

![width variety across 6 random seeds, thin and thick scratches both occur](images/scratch_width_variety.jpg)

**Delivered this session:**
- `src/soiling/effects.py` — `add_scratch()` / `generate_scratch_mask()`, the finalized
  procedural scratch generator.
- `tests/test_effects.py` — 6 fast unit tests (mask shape/range, non-empty, discontinuity,
  image actually changes, seeded reproducibility, zero-scratches no-op).

**Not yet done (next session):** port `physical_lens_soiling`'s `dirt`/`water` effects
into `src/soiling/effects.py` alongside `add_scratch`, build the MIO-TCD-based dataset
pipeline (`src/soiling/dataset_builder.py`), and the Stage A model/training/eval code —
per the approved plan.

---

## Session 2 — MIO-TCD pilot subset (Step 1)

**Date:** 2026-08-23

Downloaded `MIO-TCD-Localization.tar` (official archive, CC BY-NC-SA 4.0, 3.5 GB,
137,743 full-frame traffic-camera images) directly from the dataset provider. The archive
only offers a full-dataset download — no partial/per-image API — so the workflow is:
download once, then sample locally.

**Archive layout:** `train/` (110,000 images, has public ground truth in `gt_train.csv`,
one row per object: `image_id,class,x1,y1,x2,y2`) and `test/` (27,743 images, labels held
out for the competition, not usable). Sampled from `train/` specifically so the same pilot
subset stays reusable for any later detection-related work, even though Stage A itself
doesn't need the boxes.

**Pilot size:** 1,000 images (~1% of the train split) — matches the project's own
pilot-first philosophy (setup.md), and each source image will yield several synthetic
distortion variants in the next step, so 1,000 sources is enough to produce a few thousand
training examples for Stage A without processing the full archive.

**Delivered this session:**
- `src/data/mio_tcd.py` — pure, testable functions (`list_train_image_ids`,
  `sample_image_ids`, `filter_gt_rows`, `extract_images`) plus `build_pilot_subset()`
  tying them together: deterministic random sample of image ids (seeded) → extract just
  those images from the tar (not the whole archive) → filter the matching `gt_train.csv`
  rows → write a manifest of the sampled ids for reproducibility.
- `scripts/sample_mio_tcd.py` — CLI wrapper (`--tar`, `--out`, `--n`, `--seed`).
- `tests/test_mio_tcd.py` — 6 tests against a small synthetic in-memory tar (no real
  MIO-TCD download needed to run the suite).
- Ran it for real: `python scripts/sample_mio_tcd.py --tar D:/MasterThesis_FAU/MIO-TCD-Localization.tar --out data/raw/mio_tcd --n 1000 --seed 0`
  → 1,000 images (27 MB), 3,168 matching ground-truth rows, seed=0 manifest. Verified a
  random sample of the extracted images decodes correctly and looks like a diverse mix of
  cameras/scenes. The extracted images themselves are not committed (`data/` is
  gitignored) — only the sampling code and the seed are, so the exact same 1,000-image
  subset is reproducible by anyone with the archive.

**Not yet done (next session):** the synthetic distortion pipeline (`dirt`/`water` ported
from `physical_lens_soiling` + the existing `add_scratch`) applied to this 1,000-image
subset to build the actual Stage A multi-label dataset.

---

## Session 3 — Vendoring physical_lens_soiling; `{dirt, water, scratch}` taxonomy complete

**Date:** 2026-08-23

The supervisor cleared direct use of `physical_lens_soiling`'s code (it ships no LICENSE
file upstream, so this was a real permission, not an assumption). That changes Session 1's
plan from "reimplement the technique ourselves" to vendoring the actual source.

**What was vendored**, into `third_party/physical_lens_soiling/` (kept out of `src/` to
mark it clearly as external code, not ours):
- `add_mud.py` — `add_mudByTxture` (mud → our `dirt`), `add_dirtwaterByTxture` /
  `add_dirtwaterByTxture_slight` (water_thick / water_thin → our `water`, blob-blend).
- `add_droplet_distort.py` — `add_distort` (→ also `water`, but a physically different
  mechanism: an actual optical refraction/warp rather than a color blend).
- `generate_texture_paper.py` — the procedural texture generator both of the above depend on.

Deliberately **not** vendored (outside architecture.md's 3-class taxonomy, per this
session's earlier decision to drop them): `add_sun_glare.py`, `add_lensdust.py`, and their
own Albumentations-based comparison baseline.

**One real edit to the vendored files** (documented in
`third_party/physical_lens_soiling/NOTICE.md`): `add_mud.py` and `add_droplet_distort.py`
both had top-level script code that ran unconditionally on `import` (looping over a local
`image/` folder; one line was even `os.makedirs(..., exist_ok=False)`, which would crash on
a second import). Wrapped that code in `if __name__ == "__main__":` — no algorithmic
change, verified by running all four functions before/after and confirming identical
output.

**Wired into `src/soiling/effects.py`** as `add_dirt()` and `add_water()`, matching
`add_scratch`'s `(image, seed=None) -> (distorted_image, mask)` contract:
- `add_dirt`: picks a random texture style, calls `add_mudByTxture`.
- `add_water`: randomly picks one of three mechanisms per call — `'thick'`, `'thin'`
  (blob-blend), or `'droplet'` (optical warp) — so the water class gets real mechanism
  diversity, not just opacity variants of the same blend.

All three distortion classes now exist behind one consistent interface:

![dirt, water (all 3 mechanisms), and scratch on a real MIO-TCD image](images/dirt_water_scratch_taxonomy.jpg)

**Delivered this session:**
- `third_party/physical_lens_soiling/` — vendored source + `NOTICE.md` (provenance, what
  was kept/dropped, the exact edit made) + `UPSTREAM_README.md`.
- `requirements.txt` — added at the repo root (didn't exist before); includes
  `pythonperlin`, the one new dependency the vendored code needs.
- `src/soiling/effects.py` — `add_dirt()`, `add_water()` added alongside `add_scratch()`.
- `tests/test_effects_dirt_water.py` — 5 tests (image changes, mask shape/range, seeded
  reproducibility, all three water mechanisms exercised).
- Full suite: 17 passed.

**Not yet done (next session):** `src/soiling/dataset_builder.py` — apply `add_dirt` /
`add_water` / `add_scratch` in random combination (including "clean" negatives) across the
1,000-image MIO-TCD pilot subset to produce the actual Stage A multi-label training set.

---

## Session 4 — Taxonomy check against the paper's own figure; scratch made harsher

**Date:** 2026-08-23

Checked `dirt`/`water`/`scratch` against the physical_lens_soiling paper's own Figure 1
(mud stain, flare, dust, water mist, water droplet, water stain) instead of inventing new
mechanisms — a "residue" dirt-stain variant had been considered and was dropped in favor of
matching the paper exactly.

**Decision: no invented variants.** `add_dirt`/`add_water` stay exactly as wired in Session
3 and already match the figure one-for-one (mud stain → (b)/(c), water droplet → (g), water
stain heavy/light → (h)/(i); "water mist" (f) is just `stain (light)` drawing a fog texture,
already covered by the existing texture pool). Flare and dust stay excluded (outside
architecture.md's 3-class taxonomy).

**Scratch made harsher** per feedback that it needed to read more clearly: opacity floor
`0.35 → 0.6`, blend strength `0.85 → 0.97`.

![figure-1-style layout using only the real vendored effects, plus scratch](images/figure1_style_comparison.jpg)

**Delivered this session:** tuning only, no new modules — `src/soiling/effects.py` opacity/
blend constants adjusted. Full suite still 17/17.

---

## Session 5 — Weak mud/water outputs traced to texture-pool luck, not a code bug

**Date:** 2026-08-23

Mud and water-stain renders were barely visible, not matching the paper's figure. Explicit
instruction: do not modify `third_party/physical_lens_soiling/` — find and fix the problem
elsewhere.

**Root cause:** three of our seven texture-style choices in `_random_dirt_water_texture()`
(our own sampling code, not vendored) gave near-invisible mask coverage by construction —
most strikingly `r_water_mud`, despite its name, averaged under 13% coverage vs. ~40-51%
for the reliable modes. Earlier demo images happened to draw from that weak tail by chance.

**Fix:** narrowed `_DIRT_WATER_TEXTURE_MODS` (our sampling pool, not the vendored
algorithm) to the five modes that measure consistently strong: `r_fog`, `thick_fog`,
`f_water_mud`, `big_rain_drop`, `little_rain_drop`. No vendored file touched.

![mud/water across 3 seeds after restricting the texture pool to the reliably-strong modes](images/dirt_water_strong_texture_pool.jpg)

**Delivered this session:** `src/soiling/effects.py` — `_DIRT_WATER_TEXTURE_MODS` narrowed
from 7 to 5 entries. Full suite: 17/17.

---

## Session 6 — Dataset builder: precompute strategy, balanced single-distortion variants

**Date:** 2026-08-24

**`add_scratch` interface unified with `add_dirt`/`add_water`.** It previously took
explicit severity kwargs (`n_scratches`, `length_frac`, `width_px`, `opacity`, ...); changed
to `add_scratch(image, seed=None)`, with the amount randomized internally, matching the
other two effects' style — needed so all three effects can be driven identically by the
dataset builder. `generate_scratch_mask(...)` still takes the full parameter set directly
for anyone who wants explicit control outside the dataset pipeline.

**Augmentation strategy: precompute, not on-the-fly.** Before building the dataset, we
weighed applying distortions live during training (on-the-fly, discussed vs. precomputing a
static set once) given training happens on the shared FAU HPC allocation. The vendored
effects (Perlin-texture generation, piecewise-affine warps for the droplet mechanism) are
CPU-bound and non-trivial per call; a frozen-backbone, small-head Stage A model has a cheap
GPU step, so an on-the-fly CPU pipeline risked bottlenecking the GPU on every batch, with the
added cost paid on every epoch instead of once. Decision: precompute a static dataset once,
train against fixed files. `src/soiling/dataset_builder.py` and
`scripts/build_stage_a_dataset.py` implement this — one pass over the source images writes
distorted variants + a `metadata.csv` (`path, source_id, variant_id, split, dirt, water,
scratch`) to `data/processed/stage_a/`.

**Splits are assigned per source image, not per variant** (`assign_splits`), so all variants
of the same source frame stay in the same split — otherwise near-duplicate variants of one
frame could leak across train/val/test.

**Reproducibility gap found and accepted as a known limitation.** `add_distort` (vendored,
used for the water "droplet" mechanism) draws via Python's stdlib `random.choice`, not
`np.random` — seeding only `np.random.seed()` left that draw dependent on leftover global
state. Fixed with a `_seed_all(seed)` helper in `effects.py` that seeds both `np.random` and
`random`, used in `add_dirt`/`add_water`/`generate_scratch_mask`. Deeper down, `add_dirt`
and `add_water` (the texture-based mechanisms) still aren't pixel-reproducible: the vendored
`generate_texture_paper.py` calls `pythonperlin.perlin(...)` without a `seed=` argument, and
`pythonperlin`'s own `make_grads()` then calls `np.random.seed(None)` internally — silently
re-randomizing the global RNG from OS entropy on every texture draw, from inside a
*dependency* of the vendored code, not the vendored code itself. Fixing it cleanly would mean
monkey-patching `pythonperlin.perlin` from our own code. **Decision: leave it as-is.** The
dataset is generated once and the resulting files are the artifact — a regenerated copy
isn't expected to be pixel-identical, only label-identical (same seed → same effect assigned
to each variant slot). `tests/test_dataset_builder.py`'s reproducibility test checks labels
only, with the reasoning recorded inline.

**First build was rejected: too many clean images, and combined distortions.** The initial
design sampled each of dirt/water/scratch independently per variant with `effect_prob=0.35`,
which (a) could combine two or three distortion types onto the same image, and (b) skewed
~65% of variants clean by chance (three independent 65%-fail rolls). Both were wrong for
this dataset: **exactly one distortion type per image, never combined**, and **not
overwhelmingly clean**. Replaced with deterministic balanced assignment
(`assign_variant_kinds`): for each source image, the variant slots are filled by cycling
through `("clean", "dirt", "water", "scratch")` and shuffling the order — with
`variants_per_image=4` this guarantees exactly one clean + one of each distortion per source
image, every time, rather than leaving the mix to chance.

**Delivered this session:** `src/soiling/dataset_builder.py` (new), `scripts/
build_stage_a_dataset.py` (new), `tests/test_dataset_builder.py` (new, 11 tests). Full suite:
28/28. Ran the real build against the 1000-image MIO-TCD pilot subset
(`data/raw/mio_tcd/images`, from Session 2):

| | count |
|---|---|
| total images | 4000 (1000 sources × 4 variants) |
| train / val / test | 3200 / 400 / 400 |
| dirt positive | 1000 (25.0%) |
| water positive | 1000 (25.0%) |
| scratch positive | 1000 (25.0%) |
| clean (all-zero label) | 1000 (25.0%) |

Output not committed (gitignored, under `data/`) — regenerable via `python
scripts/build_stage_a_dataset.py --source data/raw/mio_tcd/images --out
data/processed/stage_a --variants 4 --seed 0`.

**Follow-up (2026-08-24): dataset rebalanced again.** The 0.35 `effect_prob` design above
still let a variant land clean or combine distortions by chance; direct feedback ("i might
have 3/4 images clean and this shouldnt be like that", "dont, ever mix two type of
distortion. only one") replaced it with the fully deterministic
`assign_variant_kinds` scheme described above (cycle `("clean", "dirt", "water",
"scratch")`, shuffle order) and dropped `variants_per_image` from 12 → 5 → **4** (final).
Rebuilt for real against the 1000-image pilot subset: 4000 images, exactly 1000/1000/1000/
1000 (25.0% each) for clean/dirt/water/scratch, 3200/400/400 train/val/test.

---

## Session 7 — Stage A model: frozen YOLO backbone + distortion head (Step 3)

**Date:** 2026-08-24

Added `torch` and `ultralytics` (CPU-only torch locally — training happens on the FAU HPC,
never here) and downloaded COCO-pretrained `yolo11m.pt` into `weights/` (gitignored via the
existing `*.pt` rule).

**`src/models/backbone.py` — `FrozenYOLOBackbone`.** Traced Ultralytics' `YOLO('yolo11m.pt')
.model.model` (a flat 24-layer `nn.Sequential`) to find the actual P5 tap: layers 0-10
(Conv/C3k2 stages, SPPF, C2PSA) are the shared backbone, each taking input only from the
immediately preceding layer; layer 11 is the neck's first `Upsample`, where FPN/PAN fusion
begins. So layer 10 (`C2PSA`)'s output — confirmed 512 channels, stride 32 — is P5. The
module runs just those 11 layers, freezes every parameter (`requires_grad = False`), and
overrides `.train()` to always force `eval()` so batchnorm stats can't drift while only the
head is being trained (architecture.md §3 Option 1: frozen backbone).

**`src/models/distortion_head.py` — `StageADistortionHead`.** GAP + 2×FC exactly per
architecture.md §6. Returns raw logits rather than post-sigmoid probabilities: §4 specifies
`BCEWithLogitsLoss`, which applies sigmoid internally for numerical stability — an explicit
`Sigmoid` layer in the head would double-apply it. `torch.sigmoid(head(features))` gives the
per-class probabilities §2 describes; documented inline to make the reasoning explicit
rather than silently deviating from the doc's literal "...+ sigmoid" wording.

**`src/models/losses.py`.** `compute_pos_weight(metadata_csv, split=...)` reads the
dataset's own `metadata.csv` and computes the standard `#negatives / #positives` per class
(architecture.md §4: "`pos_weight` ... for class imbalance"); `build_stage_a_loss` just
wraps `BCEWithLogitsLoss(pos_weight=...)`.

**Crash found and fixed: `torch` + vendored `scikit-image` code together aborted the
process.** Running the full suite (model tests + the existing effects/dataset-builder
tests in one process) crashed with `Fatal Python error: Aborted` inside
`add_droplet_distort.py`'s `PiecewiseAffineTransform.estimate()` — a known Windows
MKL/OpenMP conflict: both `torch` and `numpy`/`scikit-image` bundle their own Intel OpenMP
runtime (`libiomp5md.dll`), and loading it twice aborts on first real use. Not a bug in our
code. Fixed with the standard `KMP_DUPLICATE_LIB_OK=TRUE` workaround, set in a new
`tests/conftest.py` (must run before either library is imported, hence conftest rather than
inline in a test file).

**Smoke-tested end-to-end** against a real image from the Session 6 dataset: `480×720`
input → P5 `[1, 512, 15, 23]` → head logits `[1, 3]`; `compute_pos_weight` on the train
split correctly returned `[3.0, 3.0, 3.0]` (matches the dataset's exact 25%-positive-per-
class balance: 3 negatives per positive); loss finite.

**Delivered this session:** `src/models/backbone.py`, `src/models/distortion_head.py`,
`src/models/losses.py` (all new), `tests/test_backbone.py`, `tests/test_distortion_head.py`,
`tests/test_losses.py` (new, 10 tests — 3 of which skip if `weights/yolo11m.pt` isn't
present locally), `tests/conftest.py` (new). `requirements.txt` — added `torch`,
`ultralytics`. Full suite: 38/38 (all running, none skipped, since the weights are already
downloaded here).

**Not yet done (next session):** Step 4 — training script (`scripts/train_stage_a.py`,
smoke-test mode only per the standing "no training from Claude" rule) combining the
backbone, head, and loss into an actual training loop over the Stage A dataset.

---

## Session 8 — Training script (Step 4)

**Date:** 2026-08-24

**`src/data/stage_a_dataset.py` — `StageADataset`.** A `torch.utils.data.Dataset` over the
`metadata.csv` + `images/` written by `dataset_builder.py`. Filters to one `split`,
preprocesses each image the same way Ultralytics' own predictor does (BGR→RGB, resize,
HWC→CHW, `/255`) so the frozen backbone sees the input format it was actually trained on.
`max_samples` truncates the row list, used by the training script's `--smoke-test` mode.

**`scripts/train_stage_a.py`.** Frozen backbone, only `StageADistortionHead`'s parameters
go to the optimizer (`torch.optim.Adam(head.parameters(), ...)`) — matches architecture.md
§3 Option 1 exactly: backbone forward runs inside `torch.no_grad()`, only the head builds a
graph. `pos_weight` for the loss is computed once from the dataset's own `metadata.csv`
(train split). CLI: `--data`, `--weights`, `--epochs`, `--batch-size`, `--lr`, `--img-size`,
`--device`, `--out` (checkpoint dir). `--smoke-test` forces 1 epoch / batch-size 2 / CPU / 8
samples per split and skips writing a checkpoint — this is the only "training" run from this
side, per the standing rule that real training happens on the user's own FAU HPC job, never
launched from here.

Ran the smoke test for real against the actual 4000-image dataset from Session 6:
```
train=8 val=8 pos_weight=[3.0, 3.0, 3.0]
epoch 1/1  train_loss=1.0679  val_loss=1.0332  (6.0s)
```
Confirms the full backbone → head → loss → optimizer.step() pipeline actually runs on real
data, not just synthetic tensors — the point of the smoke test, not a claim about learning.

**Delivered this session:** `src/data/stage_a_dataset.py`, `scripts/train_stage_a.py` (both
new), `tests/test_stage_a_dataset.py` (5 tests), `tests/test_train_stage_a.py` (1
end-to-end subprocess test, skips without local weights). Full suite: 43/43.

**Not yet done (next session):** Step 5 — FAU HPC SLURM/env scripts
(`scripts/hpc/setup_env.sh`, `scripts/hpc/train_stage_a.slurm`), then Step 6 — evaluation
script (`scripts/evaluate_stage_a.py`, per-class precision/recall/F1/AP on the held-out
test split).
