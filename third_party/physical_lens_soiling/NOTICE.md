# Vendored: physical_lens_soiling

**Source:** https://github.com/JannLi/physical_lens_soiling
**Paper:** "Procedural Generation of Lens Soiling Data via Physics-based Simulation"
**Fetched:** 2026-08-23 (git clone, depth 1)

overview.md names this repo as the sanctioned synthetic-data source for `dirt`/`water`.
Usage cleared directly with the supervisor — the upstream repo ships no LICENSE file, so
this is used under that explicit permission rather than an open-source license grant.

## Files kept

Only the files needed for `dirt`/`water` generation (architecture.md §2's three-class
taxonomy) were brought in, from `Procudural_Generation_Lens_Soiling/`:

- `add_mud.py` — `add_mudByTxture` (mud → our `dirt`), `add_dirtwaterByTxture` /
  `add_dirtwaterByTxture_slight` (water_thick / water_thin → our `water`, blob-blend variant).
- `add_droplet_distort.py` — `add_distort` (→ our `water`, optical-refraction-warp variant;
  a second, physically different water mechanism from the blob blend above).
- `generate_texture_paper.py` — procedural texture generator both files above depend on.

**Not brought in** (out of this project's `{dirt, water, scratch}` scope): `add_sun_glare.py`
(lighting artifact, not soiling), `add_lensdust.py` (haze effect), and the
`lens_dirt_by_augmentation/` folder (their own comparison baseline against the
Albumentations library, not part of their procedural method).

## Changes made to the vendored files

No algorithmic changes. The only edit: both `add_mud.py` and `add_droplet_distort.py` had
top-level script code (looping over a local `image/` folder, writing to `image_mud/` etc.)
that ran unconditionally on `import` — including one `os.makedirs(..., exist_ok=False)`
that would raise on a second import. That script code is now wrapped in
`if __name__ == "__main__":` so the functions can be imported safely from
`src/soiling/effects.py`; the code inside those blocks is untouched. `generate_texture_paper.py`
already guarded its script section and needed no change.

## Dependency

`pythonperlin` (see `requirements.txt` at the repo root) — not needed anywhere else in
this project.
