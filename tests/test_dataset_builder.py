import csv
import json

import cv2 as cv
import numpy as np
import pytest

from src.soiling.dataset_builder import (
    ALL_VARIANT_KINDS,
    ALL_VARIANT_KINDS_WITH_SEVERITY,
    COMBO_KINDS,
    EFFECT_NAMES,
    SEVERITY_VARIANT_KINDS,
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
from src.soiling.effects import SEVERITY_LEVELS
from src.soiling.tile_labels import rasterize_tile_label


def _write_fake_sources(dir_path, n=6, size=(64, 96)):
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
        cv.imwrite(str(dir_path / f"{i:08d}.jpg"), img)


def test_derive_seed_is_deterministic_and_sensitive_to_each_part():
    a = derive_seed(0, "img1", 3, "dirt")
    b = derive_seed(0, "img1", 3, "dirt")
    c = derive_seed(0, "img1", 3, "water")
    d = derive_seed(1, "img1", 3, "dirt")
    assert a == b
    assert a != c
    assert a != d


def test_assign_variant_kinds_is_balanced_and_deterministic():
    kinds = assign_variant_kinds(4, seed=0)
    assert sorted(kinds) == sorted(VARIANT_KINDS)  # exactly one clean + one of each effect

    kinds_again = assign_variant_kinds(4, seed=0)
    assert kinds == kinds_again  # same seed -> same order

    kinds_diff_seed = assign_variant_kinds(4, seed=1)
    assert kinds_diff_seed != kinds  # different seed -> (very likely) different order

    kinds_8 = assign_variant_kinds(8, seed=0)
    assert sorted(kinds_8) == sorted(VARIANT_KINDS * 2)  # cycles to stay balanced past 4


def test_assign_variant_kinds_with_combos_is_balanced_and_deterministic():
    # Session 19+: kinds=ALL_VARIANT_KINDS (8 kinds: clean + 3 single + 3
    # pair + 1 triple) -- default (kinds=None) is untouched, this is purely
    # additive/opt-in.
    kinds = assign_variant_kinds(8, seed=0, kinds=ALL_VARIANT_KINDS)
    assert sorted(kinds) == sorted(ALL_VARIANT_KINDS)  # exactly one of each of the 8 kinds

    kinds_again = assign_variant_kinds(8, seed=0, kinds=ALL_VARIANT_KINDS)
    assert kinds == kinds_again  # same seed -> same order

    kinds_16 = assign_variant_kinds(16, seed=0, kinds=ALL_VARIANT_KINDS)
    assert sorted(kinds_16) == sorted(ALL_VARIANT_KINDS * 2)  # cycles to stay balanced past 8


def test_assign_variant_kinds_with_severity_is_balanced_and_deterministic():
    # Session 20 Round 3: kinds=VARIANT_KINDS_WITH_SEVERITY (10 kinds: clean
    # + 3 effects x 3 severity levels) -- exactly one of each kind per
    # source image gives an exact 1/3-1/3-1/3 balance per class.
    kinds = assign_variant_kinds(10, seed=0, kinds=VARIANT_KINDS_WITH_SEVERITY)
    assert sorted(kinds) == sorted(VARIANT_KINDS_WITH_SEVERITY)

    kinds_again = assign_variant_kinds(10, seed=0, kinds=VARIANT_KINDS_WITH_SEVERITY)
    assert kinds == kinds_again  # same seed -> same order

    kinds_14 = assign_variant_kinds(14, seed=0, kinds=ALL_VARIANT_KINDS_WITH_SEVERITY)
    assert sorted(kinds_14) == sorted(ALL_VARIANT_KINDS_WITH_SEVERITY)  # + the 4 combo kinds


def test_combo_from_kind_bare_single_effect_defaults_to_high():
    # A bare single-token kind (plain VARIANT_KINDS' "dirt", used when
    # include_severity=False) must default straight to "high" -- today's
    # original unparametrized behavior -- NOT get randomized like an
    # un-annotated token inside a genuine "+"-joined combo does.
    combo, severities = _combo_from_kind("dirt")
    assert combo == (True, False, False)
    assert severities == {"dirt": "high"}


def test_combo_from_kind_parses_severity_variant_kinds():
    combo, severities = _combo_from_kind("dirt:low")
    assert combo == (True, False, False)
    assert severities == {"dirt": "low"}

    combo, severities = _combo_from_kind("dirt:low+water:medium")
    assert combo == (True, True, False)
    assert severities == {"dirt": "low", "water": "medium"}


def test_combo_from_kind_plain_combo_kind_has_no_literal_severity():
    # a plain COMBO_KINDS string ("dirt+water") carries no ":level" suffix
    # -- severity is left unresolved (None) for the caller to pick.
    combo, severities = _combo_from_kind("dirt+water")
    assert combo == (True, True, False)
    assert severities == {"dirt": None, "water": None}


def test_resolve_severities_fills_in_none_deterministically():
    combo, resolved = _resolve_severities("dirt+water", base_seed=0, source_id="00000001", variant_idx=0)
    assert combo == (True, True, False)
    assert resolved["dirt"] in SEVERITY_LEVELS
    assert resolved["water"] in SEVERITY_LEVELS

    combo_again, resolved_again = _resolve_severities("dirt+water", base_seed=0, source_id="00000001", variant_idx=0)
    assert resolved == resolved_again  # deterministic given the same inputs


def test_resolve_severities_respects_explicit_levels():
    combo, resolved = _resolve_severities("dirt:low", base_seed=0, source_id="00000001", variant_idx=0)
    assert combo == (True, False, False)
    assert resolved == {"dirt": "low"}


def test_apply_effect_combo_labels_match_what_was_applied():
    img = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels = apply_effect_combo(
        img, (True, False, True), {"dirt": 1, "water": 2, "scratch": 3}
    )
    # no severities passed -> every active class defaults to "high" (today's
    # unparametrized full-strength behavior)
    assert labels == {
        "dirt": 1, "water": 0, "scratch": 1,
        "dirt_severity": "high", "water_severity": "none", "scratch_severity": "high",
    }
    assert not np.array_equal(out, img)


def test_apply_effect_combo_all_false_leaves_image_unchanged():
    img = np.random.default_rng(2).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels = apply_effect_combo(img, (False, False, False), {})
    assert labels == {
        "dirt": 0, "water": 0, "scratch": 0,
        "dirt_severity": "none", "water_severity": "none", "scratch_severity": "none",
    }
    assert np.array_equal(out, img)


def test_build_variant_is_reproducible():
    # kind="scratch" (not "dirt"/"water"): those two route through the
    # vendored texture pipeline, which has the known pythonperlin
    # np.random.seed(None) reproducibility gap (see
    # test_build_stage_a_dataset_labels_are_reproducible) -- scratch doesn't
    # use textures, so it's genuinely pixel-reproducible.
    img = np.random.default_rng(3).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out1, labels1 = build_variant(img, source_id="00000001", variant_idx=2, kind="scratch", base_seed=0)
    out2, labels2 = build_variant(img, source_id="00000001", variant_idx=2, kind="scratch", base_seed=0)
    assert np.array_equal(out1, out2)
    assert labels1 == labels2


def test_build_variant_kind_selects_exactly_one_effect():
    img = np.random.default_rng(4).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)

    out_clean, labels_clean = build_variant(img, source_id="00000001", variant_idx=0, kind="clean", base_seed=0)
    assert labels_clean == {
        "dirt": 0, "water": 0, "scratch": 0,
        "dirt_severity": "none", "water_severity": "none", "scratch_severity": "none",
    }
    assert np.array_equal(out_clean, img)

    out_water, labels_water = build_variant(img, source_id="00000001", variant_idx=0, kind="water", base_seed=0)
    # kind="water" (no explicit ":level") -- severity is resolved pseudo-
    # randomly (see _resolve_severities), so only check the class-active
    # part exactly; the severity itself just needs to be a valid level.
    assert {k: v for k, v in labels_water.items() if k in EFFECT_NAMES} == {"dirt": 0, "water": 1, "scratch": 0}
    assert labels_water["dirt_severity"] == "none"
    assert labels_water["water_severity"] in SEVERITY_LEVELS
    assert labels_water["scratch_severity"] == "none"
    assert not np.array_equal(out_water, img)


def test_build_variant_kind_selects_multiple_effects_for_combo_kind():
    # Session 19+: a "+"-joined kind string (a COMBO_KINDS member) applies
    # more than one effect to the same image -- the additive counterpart to
    # test_build_variant_kind_selects_exactly_one_effect above.
    img = np.random.default_rng(4).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)

    out, labels = build_variant(img, source_id="00000001", variant_idx=0, kind="dirt+water", base_seed=0)
    assert {k: v for k, v in labels.items() if k in EFFECT_NAMES} == {"dirt": 1, "water": 1, "scratch": 0}
    assert labels["dirt_severity"] in SEVERITY_LEVELS
    assert labels["water_severity"] in SEVERITY_LEVELS
    assert labels["scratch_severity"] == "none"
    assert not np.array_equal(out, img)

    out_triple, labels_triple = build_variant(
        img, source_id="00000001", variant_idx=0, kind="dirt+water+scratch", base_seed=0
    )
    assert {k: v for k, v in labels_triple.items() if k in EFFECT_NAMES} == {"dirt": 1, "water": 1, "scratch": 1}
    for name in EFFECT_NAMES:
        assert labels_triple[f"{name}_severity"] in SEVERITY_LEVELS
    assert not np.array_equal(out_triple, img)


def test_build_variant_differs_across_variant_index_for_same_kind():
    img = np.random.default_rng(5).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out1, _ = build_variant(img, source_id="00000001", variant_idx=0, kind="water", base_seed=0)
    out2, _ = build_variant(img, source_id="00000001", variant_idx=1, kind="water", base_seed=0)
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
        assert row["dirt"] + row["water"] + row["scratch"] <= 1

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
    assert set(csv_rows[0].keys()) == {
        "path", "source_id", "variant_id", "split", "dirt", "water", "scratch",
        "dirt_severity", "water_severity", "scratch_severity",
    }


def test_build_stage_a_dataset_variants_are_balanced_per_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(source_dir, out_dir, variants_per_image=4, seed=0)

    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)

    for source_rows in by_source.values():
        assert len(source_rows) == 4
        clean_count = sum(1 for r in source_rows if r["dirt"] + r["water"] + r["scratch"] == 0)
        assert clean_count == 1
        for name in ("dirt", "water", "scratch"):
            assert sum(r[name] for r in source_rows) == 1


def test_build_stage_a_dataset_with_combos_end_to_end(tmp_path):
    # Session 19+: include_combos=True adds the 4 combo kinds alongside the
    # original 4 -- additive check, mirrors test_build_stage_a_dataset_end_to_end
    # but for the 8-kind pool. variants_per_image=8 gives exact balance.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=3)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(
        source_dir, out_dir, variants_per_image=8, seed=0,
        ratios=(0.7, 0.15, 0.15), include_combos=True,
    )

    assert len(rows) == 3 * 8
    by_positives = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in rows:
        n_positive = row["dirt"] + row["water"] + row["scratch"]
        by_positives[n_positive] += 1
        assert (out_dir / row["path"]).exists()

    # per source image: 1 clean (0 positives), 3 singles (1 positive each),
    # 3 pairs (2 positives each), 1 triple (3 positives) -- so across 3
    # sources: 3 clean, 9 single, 9 pair, 3 triple.
    assert by_positives[0] == 3
    assert by_positives[1] == 9
    assert by_positives[2] == 9
    assert by_positives[3] == 3


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
        assert row["dirt"] + row["water"] + row["scratch"] <= 1


def test_build_stage_a_dataset_with_severity_end_to_end(tmp_path):
    # Session 20 Round 3: include_severity=True -- 10 kinds (clean + 3
    # effects x 3 levels), variants_per_image=10 gives exact balance: for
    # 4 source images, each single-effect severity kind gets exactly 4 rows.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows = build_stage_a_dataset(
        source_dir, out_dir, variants_per_image=10, seed=0, include_severity=True,
    )

    assert len(rows) == 4 * 10
    for row in rows:
        # still single-distortion-or-clean (include_combos not set)
        assert row["dirt"] + row["water"] + row["scratch"] <= 1

    for name in EFFECT_NAMES:
        by_level = {level: 0 for level in SEVERITY_LEVELS}
        for row in rows:
            if row[name] == 1:
                by_level[row[f"{name}_severity"]] += 1
        assert by_level == {"low": 4, "medium": 4, "high": 4}  # exact 1/3-1/3-1/3 balance

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert set(csv_rows[0].keys()) == {
        "path", "source_id", "variant_id", "split", "dirt", "water", "scratch",
        "dirt_severity", "water_severity", "scratch_severity",
    }


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

    labels1 = [(r["path"], r["dirt"], r["water"], r["scratch"]) for r in rows1]
    labels2 = [(r["path"], r["dirt"], r["water"], r["scratch"]) for r in rows2]
    assert labels1 == labels2


# --- Stage B (tile-grid) -----------------------------------------------


def test_build_stage_b_dataset_with_severity_reduces_tile_coverage(tmp_path):
    # Session 20 Round 3: a low-severity variant's tile-positive count
    # should be <= its high-severity counterpart's for the same effect
    # (mask scaled down by apply_severity -> fewer/no tiles clear the
    # rasterization threshold).
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=10, seed=0, img_size=64, include_severity=True,
    )

    assert len(rows) == 4 * 10
    for name in EFFECT_NAMES:
        i = EFFECT_NAMES.index(name)
        low_counts = [g[i].sum() for r, g in zip(rows, tile_labels) if r.get(f"{name}_severity") == "low"]
        high_counts = [g[i].sum() for r, g in zip(rows, tile_labels) if r.get(f"{name}_severity") == "high"]
        assert low_counts and high_counts
        assert sum(low_counts) <= sum(high_counts)


def test_build_stage_b_dataset_end_to_end(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_fake_sources(source_dir, n=4)

    out_dir = tmp_path / "out"
    rows, tile_labels = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=4, seed=0, img_size=64
    )

    assert len(rows) == 4 * 4
    assert tile_labels.shape == (len(rows), 3, 2, 2)  # img_size=64 -> grid 64//32=2
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
    clean_rows = [(r, g) for r, g in zip(rows, tile_labels) if r["dirt"] + r["water"] + r["scratch"] == 0]
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
    combo_rows = [(r, g) for r, g in zip(rows, tile_labels) if r["dirt"] + r["water"] + r["scratch"] >= 2]
    assert combo_rows  # sanity: the fixture actually produced combo variants

    for row, grid in zip(rows, tile_labels):
        for i, name in enumerate(EFFECT_NAMES):
            if row[name] == 0:
                assert grid[i].sum() == 0, f"{name} inactive but tile grid has a positive tile"


def test_apply_effect_combo_with_masks_gives_zero_mask_for_inactive_classes():
    from src.soiling.dataset_builder import apply_effect_combo_with_masks

    img = np.random.default_rng(6).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels, masks = apply_effect_combo_with_masks(
        img, (False, False, True), {"scratch": 7}
    )
    assert labels == {
        "dirt": 0, "water": 0, "scratch": 1,
        "dirt_severity": "none", "water_severity": "none", "scratch_severity": "high",
    }
    assert masks["dirt"].shape == img.shape[:2]
    assert masks["dirt"].sum() == 0
    assert masks["water"].sum() == 0
    assert masks["scratch"].sum() > 0
    assert not np.array_equal(out, img)
