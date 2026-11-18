import csv
from collections import Counter
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import (
    ALL_VARIANT_KINDS,
    ALL_VARIANT_KINDS_WITH_SEVERITY,
    EFFECT_NAMES,
    VARIANT_KINDS,
    VARIANT_KINDS_WITH_SEVERITY,
    _combo_from_kind,
    _resolve_severities,
    apply_effect_combo,
    assign_splits,
    assign_variant_kinds,
    build_stage_a_dataset,
    build_stage_b_dataset,
    build_variant,
    build_variant_with_mask,
    derive_seed,
)
from src.soiling.dataset_builder import COMBO_KINDS
from src.soiling.effects import SEVERITY_LEVELS
from src.soiling.tile_labels import rasterize_tile_label

# The class list grew from 3 to the protocol's 6 in Session 31, so these tests
# pick their classes out of EFFECT_NAMES instead of naming them, and count
# kinds with len() -- they then hold for whatever the class list is.
FIRST, SECOND, SCRATCH = EFFECT_NAMES[0], EFFECT_NAMES[1], "scratches"


def _hot(*names):
    """The multi-hot (one bool per class, in EFFECT_NAMES order) for `names`."""
    return tuple(name in names for name in EFFECT_NAMES)


def _expected_labels(**severities):
    """The labels dict for the given active classes: 1 and the severity for
    each named class, 0 and "none" for every other."""
    labels = {}
    for name in EFFECT_NAMES:
        labels[name] = int(name in severities)
        labels[f"{name}_severity"] = severities.get(name, "none")
    return labels


def _n_positive(row):
    return sum(int(row[name]) for name in EFFECT_NAMES)


def _metadata_keys():
    return {"path", "source_id", "variant_id", "split",
            *EFFECT_NAMES, *[f"{n}_severity" for n in EFFECT_NAMES]}


def _write_fake_sources(dir_path, n=6, size=(64, 96)):
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
        cv.imwrite(str(dir_path / f"{i:08d}.jpg"), img)


def test_derive_seed_is_deterministic_and_sensitive_to_each_part():
    a = derive_seed(0, "img1", 3, FIRST)
    b = derive_seed(0, "img1", 3, FIRST)
    c = derive_seed(0, "img1", 3, SECOND)
    d = derive_seed(1, "img1", 3, FIRST)
    assert a == b
    assert a != c
    assert a != d


def test_assign_variant_kinds_is_balanced_and_deterministic():
    n = len(VARIANT_KINDS)
    kinds = assign_variant_kinds(n, seed=0)
    assert sorted(kinds) == sorted(VARIANT_KINDS)  # exactly one clean + one of each effect

    kinds_again = assign_variant_kinds(n, seed=0)
    assert kinds == kinds_again  # same seed -> same order

    kinds_diff_seed = assign_variant_kinds(n, seed=1)
    assert kinds_diff_seed != kinds  # different seed -> (very likely) different order

    kinds_twice = assign_variant_kinds(2 * n, seed=0)
    assert sorted(kinds_twice) == sorted(VARIANT_KINDS * 2)  # cycles to stay balanced


def test_assign_variant_kinds_with_combos_is_balanced_and_deterministic():
    # Session 19+: kinds=ALL_VARIANT_KINDS (clean + every single type + every
    # two-type combination) -- default (kinds=None) is untouched, this is
    # purely additive/opt-in.
    n = len(ALL_VARIANT_KINDS)
    kinds = assign_variant_kinds(n, seed=0, kinds=ALL_VARIANT_KINDS)
    assert sorted(kinds) == sorted(ALL_VARIANT_KINDS)  # exactly one of each kind

    kinds_again = assign_variant_kinds(n, seed=0, kinds=ALL_VARIANT_KINDS)
    assert kinds == kinds_again  # same seed -> same order

    kinds_twice = assign_variant_kinds(2 * n, seed=0, kinds=ALL_VARIANT_KINDS)
    assert sorted(kinds_twice) == sorted(ALL_VARIANT_KINDS * 2)  # cycles to stay balanced


def test_assign_variant_kinds_with_severity_is_balanced_and_deterministic():
    # Session 20 Round 3: kinds=VARIANT_KINDS_WITH_SEVERITY (clean + every
    # type x 3 severity levels) -- exactly one of each kind per source image
    # gives an exact 1/3-1/3-1/3 balance per class.
    n = len(VARIANT_KINDS_WITH_SEVERITY)
    kinds = assign_variant_kinds(n, seed=0, kinds=VARIANT_KINDS_WITH_SEVERITY)
    assert sorted(kinds) == sorted(VARIANT_KINDS_WITH_SEVERITY)

    kinds_again = assign_variant_kinds(n, seed=0, kinds=VARIANT_KINDS_WITH_SEVERITY)
    assert kinds == kinds_again  # same seed -> same order

    all_n = len(ALL_VARIANT_KINDS_WITH_SEVERITY)
    kinds_all = assign_variant_kinds(all_n, seed=0, kinds=ALL_VARIANT_KINDS_WITH_SEVERITY)
    assert sorted(kinds_all) == sorted(ALL_VARIANT_KINDS_WITH_SEVERITY)  # + the combo kinds


def test_combo_from_kind_bare_single_effect_defaults_to_high():
    # A bare single-token kind (a plain VARIANT_KINDS type name, used when
    # include_severity=False) must default straight to "high" -- today's
    # original unparametrized behavior -- NOT get randomized like an
    # un-annotated token inside a genuine "+"-joined combo does.
    combo, severities = _combo_from_kind(FIRST)
    assert combo == _hot(FIRST)
    assert severities == {FIRST: "high"}


def test_combo_from_kind_parses_severity_variant_kinds():
    combo, severities = _combo_from_kind(f"{FIRST}:low")
    assert combo == _hot(FIRST)
    assert severities == {FIRST: "low"}

    combo, severities = _combo_from_kind(f"{FIRST}:low+{SECOND}:medium")
    assert combo == _hot(FIRST, SECOND)
    assert severities == {FIRST: "low", SECOND: "medium"}


def test_combo_from_kind_plain_combo_kind_has_no_literal_severity():
    # a plain COMBO_KINDS string carries no ":level" suffix -- severity is
    # left unresolved (None) for the caller to pick.
    combo, severities = _combo_from_kind(f"{FIRST}+{SECOND}")
    assert combo == _hot(FIRST, SECOND)
    assert severities == {FIRST: None, SECOND: None}


def test_resolve_severities_fills_in_none_deterministically():
    kind = f"{FIRST}+{SECOND}"
    combo, resolved = _resolve_severities(kind, base_seed=0, source_id="00000001", variant_idx=0)
    assert combo == _hot(FIRST, SECOND)
    assert resolved[FIRST] in SEVERITY_LEVELS
    assert resolved[SECOND] in SEVERITY_LEVELS

    combo_again, resolved_again = _resolve_severities(kind, base_seed=0, source_id="00000001", variant_idx=0)
    assert resolved == resolved_again  # deterministic given the same inputs


def test_resolve_severities_respects_explicit_levels():
    combo, resolved = _resolve_severities(f"{FIRST}:low", base_seed=0, source_id="00000001", variant_idx=0)
    assert combo == _hot(FIRST)
    assert resolved == {FIRST: "low"}


def test_apply_effect_combo_labels_match_what_was_applied():
    img = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    seeds = {name: i + 1 for i, name in enumerate(EFFECT_NAMES)}
    out, labels = apply_effect_combo(img, _hot(FIRST, SCRATCH), seeds)
    # no severities passed -> every active class defaults to "high" (today's
    # unparametrized full-strength behavior)
    assert labels == _expected_labels(**{FIRST: "high", SCRATCH: "high"})
    assert not np.array_equal(out, img)


def test_apply_effect_combo_all_false_leaves_image_unchanged():
    img = np.random.default_rng(2).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels = apply_effect_combo(img, _hot(), {})
    assert labels == _expected_labels()
    assert np.array_equal(out, img)


def test_build_variant_is_reproducible():
    # kind=SCRATCH (not an area effect): those route through the
    # vendored texture pipeline, which has the known pythonperlin
    # np.random.seed(None) reproducibility gap (see
    # test_build_stage_a_dataset_labels_are_reproducible) -- scratch doesn't
    # use textures, so it's genuinely pixel-reproducible.
    img = np.random.default_rng(3).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out1, labels1 = build_variant(img, source_id="00000001", variant_idx=2, kind=SCRATCH, base_seed=0)
    out2, labels2 = build_variant(img, source_id="00000001", variant_idx=2, kind=SCRATCH, base_seed=0)
    assert np.array_equal(out1, out2)
    assert labels1 == labels2


def test_build_variant_kind_selects_exactly_one_effect():
    img = np.random.default_rng(4).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)

    out_clean, labels_clean = build_variant(img, source_id="00000001", variant_idx=0, kind="clean", base_seed=0)
    assert labels_clean == _expected_labels()
    assert np.array_equal(out_clean, img)

    out_one, labels_one = build_variant(img, source_id="00000001", variant_idx=0, kind=SCRATCH, base_seed=0)
    # a bare type name (no explicit ":level") -- severity is resolved pseudo-
    # randomly (see _resolve_severities), so only check the class-active
    # part exactly; the severity itself just needs to be a valid level.
    assert {k: v for k, v in labels_one.items() if k in EFFECT_NAMES} == \
        {name: int(name == SCRATCH) for name in EFFECT_NAMES}
    assert labels_one[f"{SCRATCH}_severity"] in SEVERITY_LEVELS
    for name in EFFECT_NAMES:
        if name != SCRATCH:
            assert labels_one[f"{name}_severity"] == "none"
    assert not np.array_equal(out_one, img)


def test_build_variant_kind_selects_multiple_effects_for_combo_kind():
    # Session 19+: a "+"-joined kind string (a COMBO_KINDS member) applies
    # more than one effect to the same image -- the additive counterpart to
    # test_build_variant_kind_selects_exactly_one_effect above.
    img = np.random.default_rng(4).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)

    pair = f"{SCRATCH}+{SECOND}"
    out, labels = build_variant(img, source_id="00000001", variant_idx=0, kind=pair, base_seed=0)
    assert {k: v for k, v in labels.items() if k in EFFECT_NAMES} == \
        {name: int(name in (SCRATCH, SECOND)) for name in EFFECT_NAMES}
    assert labels[f"{SCRATCH}_severity"] in SEVERITY_LEVELS
    assert labels[f"{SECOND}_severity"] in SEVERITY_LEVELS
    for name in EFFECT_NAMES:
        if name not in (SCRATCH, SECOND):
            assert labels[f"{name}_severity"] == "none"
    assert not np.array_equal(out, img)


def test_combo_kinds_are_every_pair_and_never_three_types():
    # Session 31, the supervisor's rule: a contaminated image carries exactly
    # one or exactly two types, never three (capture_plan.md).
    assert all(len(kind.split("+")) == 2 for kind in COMBO_KINDS)
    expected = {frozenset((a, b)) for a in EFFECT_NAMES for b in EFFECT_NAMES if a != b}
    assert {frozenset(kind.split("+")) for kind in COMBO_KINDS} == expected
    assert len(COMBO_KINDS) == len(expected)


def test_build_variant_differs_across_variant_index_for_same_kind():
    img = np.random.default_rng(5).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out1, _ = build_variant(img, source_id="00000001", variant_idx=0, kind=SCRATCH, base_seed=0)
    out2, _ = build_variant(img, source_id="00000001", variant_idx=1, kind=SCRATCH, base_seed=0)
    assert not np.array_equal(out1, out2)


def test_assign_splits_keeps_all_ids_and_matches_ratios():
    ids = [f"{i:08d}" for i in range(100)]
    split_of = assign_splits(ids, seed=0, ratios=(0.8, 0.1, 0.1))
    assert set(split_of) == set(ids)
    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_of.values():
        counts[s] += 1
    assert counts["train"] == 80
    assert counts["val"] == 10
    assert counts["test"] == 10


def test_build_stage_a_dataset_end_to_end(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=6)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(
        source_dir, out_dir, variants_per_image=3, seed=0, ratios=(0.7, 0.15, 0.15)
    )

    assert len(rows) == 6 * 3
    for row in rows:
        assert (out_dir / row["path"]).exists()
        # never more than one distortion type combined on a single variant
        assert _n_positive(row) <= 1

    # every variant of a given source id lands in the same split
    split_by_source = {}
    for row in rows:
        split_by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(s) == 1 for s in split_by_source.values())

    metadata_path = out_dir / "metadata.csv"
    assert metadata_path.exists()
    with open(metadata_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert set(csv_rows[0].keys()) == _metadata_keys()


def test_build_stage_a_dataset_variants_are_balanced_per_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    n_kinds = len(VARIANT_KINDS)  # clean + one per type
    rows = build_stage_a_dataset(source_dir, out_dir, variants_per_image=n_kinds, seed=0)

    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)

    for source_rows in by_source.values():
        assert len(source_rows) == n_kinds
        clean_count = sum(1 for r in source_rows if _n_positive(r) == 0)
        assert clean_count == 1
        for name in EFFECT_NAMES:
            assert sum(r[name] for r in source_rows) == 1


def test_build_stage_a_dataset_with_combos_end_to_end(tmp_path):
    # Session 19+: include_combos=True adds the two-type kinds (every pair,
    # since Session 31) alongside clean and the single types -- one variant per
    # kind gives exact balance.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=1)

    out_dir = tmp_path / "out"
    n_kinds = len(ALL_VARIANT_KINDS)
    rows = build_stage_a_dataset(
        source_dir, out_dir, variants_per_image=n_kinds, seed=0,
        ratios=(1.0, 0.0, 0.0), include_combos=True,
    )

    assert len(rows) == n_kinds
    by_positives = {0: 0, 1: 0, 2: 0}
    for row in rows:
        by_positives[_n_positive(row)] += 1  # a KeyError here would mean 3 types on one image
        assert (out_dir / row["path"]).exists()

    # 1 clean (0 positives), one per type (1 each), one per pair (2 each)
    assert by_positives[0] == 1
    assert by_positives[1] == len(EFFECT_NAMES)
    assert by_positives[2] == len(COMBO_KINDS)


def test_build_stage_a_dataset_default_still_excludes_combos(tmp_path):
    # Regression check: omitting include_combos (or passing False) must
    # still produce the original single-distortion-or-clean-only behavior --
    # the existing blanket "<=1 positive" invariant, unaffected by the new
    # combo support existing in the same function.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(source_dir, out_dir, variants_per_image=4, seed=0)
    for row in rows:
        assert _n_positive(row) <= 1


def test_build_stage_a_dataset_with_severity_end_to_end(tmp_path):
    # Session 20 Round 3: include_severity=True -- clean + every type x 3
    # levels; one variant per kind gives exact balance: for 2 source images,
    # each single-type severity kind gets exactly 2 rows.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    n_kinds = len(VARIANT_KINDS_WITH_SEVERITY)
    rows = build_stage_a_dataset(
        source_dir, out_dir, variants_per_image=n_kinds, seed=0, include_severity=True,
    )

    assert len(rows) == 2 * n_kinds
    for row in rows:
        # still single-distortion-or-clean (include_combos not set)
        assert _n_positive(row) <= 1

    for name in EFFECT_NAMES:
        by_level = {level: 0 for level in SEVERITY_LEVELS}
        for row in rows:
            if row[name] == 1:
                by_level[row[f"{name}_severity"]] += 1
        assert by_level == {"low": 2, "medium": 2, "high": 2}  # exact 1/3-1/3-1/3 balance

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert set(csv_rows[0].keys()) == _metadata_keys()


def test_build_stage_a_dataset_severity_default_off_is_always_high(tmp_path):
    # Regression check: omitting include_severity (or passing False) must
    # still make every active class "high" -- unchanged from before this
    # feature existed, just now explicit in the metadata.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(source_dir, out_dir, variants_per_image=4, seed=0)
    for row in rows:
        for name in EFFECT_NAMES:
            expected = "high" if row[name] == 1 else "none"
            assert row[f"{name}_severity"] == expected


def test_build_stage_a_dataset_labels_are_reproducible(tmp_path):
    # Labels (which effects were applied to which variant) are fully
    # reproducible for a given seed -- exact pixels are not, because
    # generate_texture_paper.py (vendored) calls pythonperlin's perlin()
    # without a seed, and pythonperlin's make_grads() then calls
    # np.random.seed(None) internally, which re-randomizes the global RNG
    # from OS entropy on every texture draw. That's inside a dependency of
    # the vendored code, not something we control without patching it, and
    # by decision it's left as-is: the dataset is generated once and the
    # files themselves are the artifact, not regenerated expecting an
    # identical copy. See docs/development_log.md.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir1 = tmp_path / "out1"
    out_dir2 = tmp_path / "out2"
    rows1 = build_stage_a_dataset(source_dir, out_dir1, variants_per_image=2, seed=0)
    rows2 = build_stage_a_dataset(source_dir, out_dir2, variants_per_image=2, seed=0)

    labels1 = [(r["path"], *[r[n] for n in EFFECT_NAMES]) for r in rows1]
    labels2 = [(r["path"], *[r[n] for n in EFFECT_NAMES]) for r in rows2]
    assert labels1 == labels2


# --- Stage B (tile-grid) -----------------------------------------------


def test_apply_effect_combo_with_masks_ground_truth_grows_with_severity():
    # Since Session 29 severity is the amount of contamination: the mask is the
    # effect's own full-strength mask, and a lighter level is part of the
    # heavier one (more sanding adds scratches). Before, severity only faded
    # one pattern and the mask was identical at every level.
    from src.soiling.dataset_builder import apply_effect_combo_with_masks

    img = np.full((96, 128, 3), 40, dtype=np.uint8)  # dark, so every level is visible on the first try
    results = {
        level: apply_effect_combo_with_masks(img, _hot(SCRATCH), {SCRATCH: 7}, {SCRATCH: level})
        for level in ("low", "medium", "high")
    }
    high_out, _, high_masks = results["high"]
    for level in ("low", "medium"):
        out, labels, masks = results[level]
        assert labels[f"{SCRATCH}_severity"] == level
        assert np.all(masks[SCRATCH] <= high_masks[SCRATCH] + 1e-6)  # contained in heavy
        assert (masks[SCRATCH] > 0.5).sum() < (high_masks[SCRATCH] > 0.5).sum()
        assert not np.array_equal(out, high_out)


def test_build_stage_b_dataset_meta_records_severity_independent_gt(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    build_stage_b_dataset(source_dir, out_dir, variants_per_image=4, seed=0, img_size=64)

    with open(out_dir / "stage_b_meta.json") as f:
        assert json.load(f)["severity_independent_gt"] is True


def test_build_stage_b_dataset_end_to_end(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64
    )

    assert len(rows) == 4 * 4
    assert tile_labels.shape == (len(rows), len(EFFECT_NAMES), 2, 2)  # img_size=64 -> grid 64//32=2
    assert tile_labels.dtype == np.uint8

    for path in ("images", "metadata.csv", "tile_labels.npy", "stage_b_meta.json"):
        assert (out_dir / path).exists()

    with open(out_dir / "stage_b_meta.json") as f:
        meta = json.load(f)
    assert meta["img_size"] == 64
    assert meta["grid_h"] == meta["grid_w"] == 2
    assert meta["class_names"] == list(EFFECT_NAMES)

    loaded = np.load(out_dir / "tile_labels.npy")
    assert np.array_equal(loaded, tile_labels)


def test_build_stage_b_dataset_inactive_classes_have_all_zero_grid(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64
    )

    for row, grid in zip(rows, tile_labels):
        for i, name in enumerate(EFFECT_NAMES):
            if row[name] == 0:
                assert grid[i].sum() == 0, f"{name} inactive but tile grid has a positive tile"

    # a clean variant (all three labels 0) must have every channel all-zero
    clean_rows = [(r, g) for r, g in zip(rows, tile_labels) if _n_positive(r) == 0]
    assert clean_rows  # sanity: the fixture actually produced clean variants
    for row, grid in clean_rows:
        assert grid.sum() == 0


def test_build_stage_b_dataset_tile_labels_match_the_mask_that_made_the_image(tmp_path):
    # Recomputes the same variant (kind, seeds) the builder used internally,
    # via the public build_variant_with_mask function, and checks the stored
    # tile label is exactly the rasterization of *that* mask -- i.e. no
    # discrepancy between the saved image and its tile label (the whole
    # point of computing them in the same call, see module docstring).
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64
    )

    with open(out_dir / "stage_b_meta.json") as f:
        meta = json.load(f)
    thresholds = meta["thresholds"]

    kind_order_seed = derive_seed(0, "00000000", "kind_order")
    kinds = assign_variant_kinds(4, kind_order_seed)
    img = cv.imread(str(source_dir / "00000000.jpg"))

    for variant_idx, kind in enumerate(kinds):
        row_idx = variant_idx  # source "00000000" is written first, variants in order
        _, _, masks = build_variant_with_mask(img, "00000000", variant_idx, kind, base_seed=0)
        expected = np.stack([
            rasterize_tile_label(masks[name], meta["grid_h"], meta["grid_w"], thresholds[name])
            for name in EFFECT_NAMES
        ])
        assert np.array_equal(tile_labels[row_idx], expected)
        assert rows[row_idx]["source_id"] == "00000000"
        assert rows[row_idx]["variant_id"] == variant_idx


def test_build_stage_b_dataset_with_combos_inactive_classes_still_all_zero(tmp_path):
    # Session 19+: include_combos=True -- a combo variant can have 2 or 3
    # classes positive in the SAME tile grid at once, wherever the combined
    # effects' masks overlap. This confirms rasterize_tile_label's existing
    # per-class independence (it was already called once per class in a
    # loop, unchanged by this feature) extends correctly: each class's
    # tile-positive rate still only depends on that class's own mask, not on
    # how many other classes are also active on the same image.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=8, seed=0, img_size=64, include_combos=True,
    )

    assert len(rows) == 3 * 8
    combo_rows = [(r, g) for r, g in zip(rows, tile_labels) if _n_positive(r) >= 2]
    assert combo_rows  # sanity: the fixture actually produced combo variants

    for row, grid in zip(rows, tile_labels):
        for i, name in enumerate(EFFECT_NAMES):
            if row[name] == 0:
                assert grid[i].sum() == 0, f"{name} inactive but tile grid has a positive tile"


def test_build_stage_b_dataset_save_pixel_masks_off_by_default(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    build_stage_b_dataset(source_dir, out_dir, variants_per_image=4, seed=0, img_size=64)

    assert not (out_dir / "masks").exists()
    with open(out_dir / "stage_b_meta.json") as f:
        meta = json.load(f)
    assert meta["has_pixel_masks"] is False


def test_build_stage_b_dataset_save_pixel_masks_writes_one_file_per_row(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2, size=(64, 96))

    out_dir = tmp_path / "out"
    rows, _ = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64, save_pixel_masks=True,
    )

    assert (out_dir / "masks").exists()
    with open(out_dir / "stage_b_meta.json") as f:
        meta = json.load(f)
    assert meta["has_pixel_masks"] is True

    for row in rows:
        mask_path = out_dir / "masks" / (Path(row["path"]).stem + ".npy")  # .npy since Session 31
        assert mask_path.exists()
        mask = np.load(mask_path)
        assert mask.shape == (64, 96, len(EFFECT_NAMES))
        assert mask.dtype == np.uint8


def test_build_stage_b_dataset_save_pixel_masks_consistent_with_saved_tile_labels(tmp_path):
    # The saved mask and the saved tile_labels grid both come from the
    # exact same in-call `masks[name]` value (see module docstring on why a
    # mask can't be pixel-reproduced by a second, separate call to
    # build_variant_with_mask -- pythonperlin silently reseeds np.random
    # internally, per tile_labels.py). This checks they stay consistent
    # WITH EACH OTHER: rasterizing the saved mask the same way the builder
    # rasterized its own in-call mask must reproduce the exact same stored
    # tile-label grid.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=2)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64, save_pixel_masks=True,
    )

    with open(out_dir / "stage_b_meta.json") as f:
        meta = json.load(f)
    thresholds = meta["thresholds"]

    for row, grid in zip(rows, tile_labels):
        saved = np.load(out_dir / "masks" / (Path(row["path"]).stem + ".npy"))
        for i, name in enumerate(EFFECT_NAMES):
            mask_float = saved[..., i].astype(np.float32) / 255.0
            expected = rasterize_tile_label(mask_float, meta["grid_h"], meta["grid_w"], thresholds[name])
            assert np.array_equal(grid[i], expected)


def test_build_stage_b_dataset_save_pixel_masks_inactive_class_is_all_zero(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir = tmp_path / "out"
    rows, _ = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64, save_pixel_masks=True,
    )

    for row in rows:
        mask = np.load(out_dir / "masks" / (Path(row["path"]).stem + ".npy"))
        for i, name in enumerate(EFFECT_NAMES):
            if row[name] == 0:
                assert mask[..., i].sum() == 0, f"{name} inactive but its mask channel is nonzero"


def test_apply_effect_combo_with_masks_gives_zero_mask_for_inactive_classes():
    from src.soiling.dataset_builder import apply_effect_combo_with_masks

    img = np.random.default_rng(6).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels, masks = apply_effect_combo_with_masks(img, _hot(SCRATCH), {SCRATCH: 7})
    assert labels == _expected_labels(**{SCRATCH: "high"})
    for name in EFFECT_NAMES:
        assert masks[name].shape == img.shape[:2]
        if name != SCRATCH:
            assert masks[name].sum() == 0
    assert masks[SCRATCH].sum() > 0
    assert not np.array_equal(out, img)


# --- the planned amounts (Session 31) ----------------------------------


def test_plan_variant_kinds_matches_the_planned_dataset():
    from src.soiling.dataset_builder import PLANNED_SINGLE_KINDS, plan_variant_kinds

    ids = [f"{i:08d}" for i in range(1000)]
    split_of = assign_splits(ids, seed=0, ratios=(0.8, 0.1, 0.1))
    plan = plan_variant_kinds(split_of, seed=0)

    assert set(plan) == set(ids)
    kinds = [kind for photo_kinds in plan.values() for kind in photo_kinds]
    counts = Counter(kinds)

    # every photo carries clean and every (type, severity) exactly once
    for kind in PLANNED_SINGLE_KINDS:
        assert counts[kind] == len(ids), kind
    assert all(set(PLANNED_SINGLE_KINDS) <= set(photo_kinds) for photo_kinds in plan.values())

    # every two-type combination is used equally often, and they are 10% of all
    assert {counts[kind] for kind in COMBO_KINDS} == {141}
    two_type = sum(counts[kind] for kind in COMBO_KINDS)
    assert two_type == 141 * len(COMBO_KINDS) == 2115
    assert len(kinds) == 21115
    assert round(two_type / len(kinds), 3) == 0.100

    # ... and equally present in each split, which is also 80/10/10 overall
    for split, per_kind, total in (("train", 113, 16895), ("val", 14, 2110), ("test", 14, 2110)):
        split_kinds = Counter(k for sid, ks in plan.items() if split_of[sid] == split for k in ks)
        assert sum(split_kinds.values()) == total
        assert {split_kinds[kind] for kind in COMBO_KINDS} == {per_kind}

    assert plan_variant_kinds(split_of, seed=0) == plan  # deterministic
    assert plan_variant_kinds(split_of, seed=1) != plan  # and seeded


def test_plan_variant_kinds_never_puts_three_types_on_one_image():
    from src.soiling.dataset_builder import plan_variant_kinds

    split_of = assign_splits([f"{i:08d}" for i in range(40)], seed=0)
    for photo_kinds in plan_variant_kinds(split_of, seed=0).values():
        for kind in photo_kinds:
            assert len(kind.split("+")) <= 2
