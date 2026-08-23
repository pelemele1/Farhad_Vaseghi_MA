import csv

import cv2 as cv
import numpy as np
import pytest

from src.soiling.dataset_builder import (
    VARIANT_KINDS,
    apply_effect_combo,
    assign_splits,
    assign_variant_kinds,
    build_stage_a_dataset,
    build_variant,
    derive_seed,
)


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


def test_apply_effect_combo_labels_match_what_was_applied():
    img = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels = apply_effect_combo(
        img, (True, False, True), {"dirt": 1, "water": 2, "scratch": 3}
    )
    assert labels == {"dirt": 1, "water": 0, "scratch": 1}
    assert not np.array_equal(out, img)


def test_apply_effect_combo_all_false_leaves_image_unchanged():
    img = np.random.default_rng(2).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    out, labels = apply_effect_combo(img, (False, False, False), {})
    assert labels == {"dirt": 0, "water": 0, "scratch": 0}
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
    assert labels_clean == {"dirt": 0, "water": 0, "scratch": 0}
    assert np.array_equal(out_clean, img)

    out_water, labels_water = build_variant(img, source_id="00000001", variant_idx=0, kind="water", base_seed=0)
    assert labels_water == {"dirt": 0, "water": 1, "scratch": 0}
    assert not np.array_equal(out_water, img)


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
        "path", "source_id", "variant_id", "split", "dirt", "water", "scratch"
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
