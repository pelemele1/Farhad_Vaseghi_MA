"""Session 27: visible-change labels, the one-dataset builder, 4-class losses,
the map metrics and the gate-first pipeline's answer."""
import csv
import json

import cv2 as cv
import numpy as np
import pytest
import torch

from src.data.visible_dataset import VisibleDataset, read_label_map
from src.eval.class_maps import (
    ap_from_histograms,
    map_stats,
    metrics_by_severity,
    metrics_from_stats,
    stopped_stats,
    sum_stats,
)
from src.models.losses import CEDiceLoss, MultiClassFocalLoss, inverse_sqrt_frequency_weights
from src.pipeline import colorize, format_answer, image_answer
from src.soiling import visible_labels
from src.soiling.effects import add_dirt, add_water
from src.soiling.visible_dataset import build_visible_dataset
from src.soiling.visible_labels import clean_up_labels, render_chain, tile_class_grid, visible_label_map

CLASSES = ("clean", "dirt", "water", "scratch")


def _photo(seed=0, h=96, w=128):
    return np.random.default_rng(seed).integers(60, 180, size=(h, w, 3), dtype=np.uint8)


# --- effects are now pixel-reproducible ----------------------------------------


@pytest.mark.parametrize("effect", [add_dirt, add_water])
def test_dirt_and_water_are_pixel_reproducible(effect):
    image = _photo(h=192, w=256)
    a, ma = effect(image, seed=11)
    b, mb = effect(image, seed=11)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_effect_shape_does_not_depend_on_image_content():
    image = _photo(h=192, w=256)
    _, m1 = add_dirt(image, seed=5)
    _, m2 = add_dirt(255 - image, seed=5)
    assert np.array_equal(m1, m2)


# --- labeling rule, with fake effects so the expected map is exact ---------------


def _fake_effects(monkeypatch):
    def dirt(img, seed=None):  # darkens the left half by 60
        out = img.astype(np.int16)
        mask = np.zeros(img.shape[:2], np.float32)
        mask[:, :64] = 1
        out[:, :64] -= 60
        return np.clip(out, 0, 255).astype(np.uint8), mask

    def water(img, seed=None):  # opaque bright film over the top 32 rows
        out = img.copy()
        mask = np.zeros(img.shape[:2], np.float32)
        mask[:32] = 1
        out[:32] = 200
        return out, mask

    def scratch(img, seed=None):  # faint bright line in row 70, across dirt and clean
        out = img.astype(np.int16)
        mask = np.zeros(img.shape[:2], np.float32)
        mask[69:72] = 1
        out[69:72] += 25
        return np.clip(out, 0, 255).astype(np.uint8), mask

    monkeypatch.setattr(visible_labels, "_EFFECTS", {"dirt": dirt, "water": water, "scratch": scratch})


def test_single_effect_label_is_where_the_image_visibly_changed(monkeypatch):
    _fake_effects(monkeypatch)
    image = np.full((96, 128, 3), 120, np.uint8)
    steps = [("dirt", 0, "high")]
    final, masks = render_chain(image, steps)
    labels = visible_label_map(image, final, steps, masks)
    assert (labels[:, :60] == 1).all()
    assert (labels[:, 68:] == 0).all()


def test_hidden_lower_layer_is_not_labeled(monkeypatch):
    _fake_effects(monkeypatch)
    image = np.full((96, 128, 3), 120, np.uint8)
    steps = [("dirt", 0, "high"), ("water", 0, "high")]
    final, masks = render_chain(image, steps)
    labels = visible_label_map(image, final, steps, masks)
    assert (labels[:28, :] == 2).all()          # water on top hides the dirt completely
    assert (labels[36:, :60] == 1).all()        # dirt where the water isn't
    assert (labels[36:, 68:] == 0).all()


def test_visible_scratch_wins_over_the_dirt_under_it(monkeypatch):
    _fake_effects(monkeypatch)
    image = np.full((96, 128, 3), 120, np.uint8)
    steps = [("dirt", 0, "high"), ("scratch", 0, "high")]
    final, masks = render_chain(image, steps)
    labels = visible_label_map(image, final, steps, masks)
    assert (labels[70, 4:124] == 3).all()       # over the dirt (change 60 < dirt's) and over clean
    assert (labels[40, :60] == 1).all()


def test_low_severity_change_below_cutoff_is_clean(monkeypatch):
    _fake_effects(monkeypatch)
    image = np.full((96, 128, 3), 120, np.uint8)
    steps = [("scratch", 0, "low")]             # 25 * 0.3 = 7.5 gray levels < 10
    final, masks = render_chain(image, steps)
    assert visible_label_map(image, final, steps, masks).max() == 0


def test_clean_up_removes_speckles_and_fills_pinholes():
    labels = np.zeros((60, 60), np.uint8)
    labels[10:40, 10:40] = 2
    labels[20, 20] = 0                          # pinhole inside water
    labels[50, 50] = 1                          # one-pixel dirt speck
    out = clean_up_labels(labels)
    assert out[20, 20] == 2 and out[50, 50] == 0
    assert (out[10:40, 10:40] == 2).all()


def test_tile_class_uses_per_class_cutoffs():
    labels = np.zeros((64, 64), np.uint8)
    labels[:32, :32][:16] = 1                   # tile (0,0): 50% dirt
    labels[:32, 32:][:4] = 1                    # tile (0,1): 12.5% dirt -> below 0.20
    labels[32:, :32][:16] = 1                   # tile (1,0): 50% dirt + a 1-px scratch line
    labels[40, :32] = 3                         # 32 px = 3.1% -> 2x the scratch cutoff, < 2.5x dirt
    grid = tile_class_grid(labels, 2, 2, {"dirt": 0.20, "water": 0.25, "scratch": 0.015})
    assert grid.tolist() == [[1, 0], [1, 0]]
    labels[41:44, :32] = 3                      # now 12.5% scratch -> 8x its cutoff
    assert tile_class_grid(labels, 2, 2, {"dirt": 0.20, "water": 0.25, "scratch": 0.015})[1, 0] == 3


# --- dataset build + loader on tiny photos --------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    for i in range(3):
        cv.imwrite(str(src / f"{i:08d}.jpg"), _photo(seed=i, h=160, w=224))
    out = tmp_path_factory.mktemp("visible")
    build_visible_dataset(src, out, variants_per_image=14, seed=0, ratios=(1 / 3, 1 / 3, 1 / 3), img_size=64)
    return out


def test_build_writes_consistent_labels(tiny_dataset):
    rows = list(csv.DictReader(open(tiny_dataset / "metadata.csv")))
    tiles = np.load(tiny_dataset / "tile_labels.npy")
    meta = json.loads((tiny_dataset / "visible_meta.json").read_text())
    assert len(rows) == 42 and tiles.shape == (42, 2, 2)
    for r in rows:
        labels = read_label_map(tiny_dataset / "labels" / (r["path"][7:-4] + ".png"))
        assert labels.shape == (160, 224) and labels.max() <= 3
        for k, name in enumerate(CLASSES[1:], start=1):
            assert int(r[f"{name}_pixels"]) == int((labels == k).sum())
            assert int(r[name]) == int(int(r[f"{name}_pixels"]) >= 20)
            assert (r[f"{name}_severity"] == "none") == (not int(r[name]))
        if r["kind"] == "clean":
            assert labels.max() == 0 and r["dominant"] == "clean"
    assert meta["class_names"] == list(CLASSES)


def test_build_is_reproducible(tiny_dataset, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    cv.imwrite(str(src / "00000000.jpg"), _photo(seed=0, h=160, w=224))
    build_visible_dataset(src, tmp_path / "again", variants_per_image=14, seed=0, ratios=(1 / 3, 1 / 3, 1 / 3),
                          img_size=64)
    for v in range(14):
        name = f"00000000_v{v:02d}.png"
        a = read_label_map(tiny_dataset / "labels" / name)
        b = read_label_map(tmp_path / "again" / "labels" / name)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("target,shape,dtype", [("image", (3,), torch.float32), ("tile", (2, 2), torch.int64),
                                                ("pixel", (64, 64), torch.int64)])
def test_visible_dataset_targets(tiny_dataset, target, shape, dtype):
    ds = VisibleDataset(tiny_dataset, "train", target, hflip=True)
    image, y = ds[0]
    assert image.shape == (3, 64, 64) and tuple(y.shape) == shape and y.dtype == dtype


# --- losses ----------------------------------------------------------------------


def test_inverse_sqrt_weights_favor_rare_classes():
    w = inverse_sqrt_frequency_weights([900, 50, 49, 1])
    assert w.mean().item() == pytest.approx(1.0) and w[3] > w[1] > w[0]


@pytest.mark.parametrize("loss_fn", [MultiClassFocalLoss(torch.ones(4)), CEDiceLoss(torch.ones(4))])
def test_four_class_losses_prefer_correct_logits(loss_fn):
    target = torch.randint(0, 4, (2, 8, 8))
    good = torch.nn.functional.one_hot(target, 4).movedim(-1, 1).float() * 10
    assert loss_fn(good, target) < loss_fn(-good, target)


# --- map metrics -------------------------------------------------------------------


def test_map_stats_and_metrics_on_a_known_case():
    gt = np.array([[0, 1], [2, 3]])
    probs = np.zeros((4, 2, 2), np.float32)
    probs[0, 0, 0] = probs[1, 0, 1] = probs[2, 1, 0] = 1.0
    probs[1, 1, 1] = 1.0                        # scratch pixel predicted dirt
    rows = metrics_from_stats(map_stats(probs, gt), CLASSES)
    by = {r["class"]: r for r in rows}
    assert by["water"]["f1"] == 1.0 and by["water"]["ap"] == pytest.approx(1.0)
    assert by["dirt"]["precision"] == 0.5 and by["dirt"]["recall"] == 1.0
    assert by["scratch"]["recall"] == 0.0


def test_stopped_stats_predict_everything_clean():
    gt = np.array([[0, 1], [2, 3]])
    s = stopped_stats(map_stats(np.full((4, 2, 2), 0.25, np.float32), gt))
    assert s["conf"][:, 1:].sum() == 0 and s["conf"][:, 0].tolist() == [1, 1, 1, 1]


def test_ap_from_histograms():
    assert ap_from_histograms(np.array([0, 0, 5]), np.array([5, 0, 0])) == pytest.approx(1.0)
    assert np.isnan(ap_from_histograms(np.zeros(3, int), np.array([1, 1, 1])))


def test_metrics_by_severity_uses_level_positives_and_all_negatives():
    gt_pos, gt_neg = np.ones((2, 2), int), np.zeros((2, 2), int)
    probs = np.zeros((4, 2, 2), np.float32)
    probs[1] = 1.0
    stats = [map_stats(probs, gt_pos), map_stats(probs, gt_neg)]
    rows = [{"dirt_severity": "low", "water_severity": "none", "scratch_severity": "none"},
            {"dirt_severity": "none", "water_severity": "none", "scratch_severity": "none"}]
    res = metrics_by_severity(rows, stats, CLASSES)
    assert res["dirt"]["low"]["recall"] == 1.0 and res["dirt"]["low"]["precision"] == 0.5
    assert sum_stats(stats)["conf"].sum() == 8


# --- pipeline answer -------------------------------------------------------------------


def test_image_answer_dominant_and_shares():
    cmap = np.zeros((100, 100), int)
    cmap[:40] = 2
    cmap[40:45] = 1
    cmap[99, 99] = 3                            # a single pixel: below MIN_SHARE
    ans = image_answer(cmap, True)
    assert ans["dominant"] == "water" and set(ans["shares"]) == {"water", "dirt"}
    assert ans["shares"]["water"] == pytest.approx(0.4)
    assert "dominant: water" in format_answer(ans)


def test_image_answer_when_gated_or_unlocated():
    clean = {"impaired": False, "dominant": "clean", "shares": {}}
    cmap = np.zeros((10, 10), int)
    assert image_answer(cmap, False) == clean
    assert image_answer(cmap, True) == clean            # gate passed, nothing located
    assert format_answer(image_answer(cmap, False)) == "clean"


def test_image_answer_per_class_min_share():
    cmap = np.zeros((100, 100), int)
    cmap[:5] = 1                                        # dirt 5%
    cmap[50, :20] = 3                                   # scratch 0.2%
    ans = image_answer(cmap, True, {"dirt": 0.1, "water": 0.01, "scratch": 0.001})
    assert ans["shares"] == {"scratch": pytest.approx(0.002)} and ans["dominant"] == "scratch"


def test_colorize_paints_each_class():
    cmap = np.array([[0, 1], [2, 3]])
    rgb = colorize(cmap)
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[1, 1].tolist() == [235, 20, 60]


def test_decision_offsets_shift_the_argmax():
    from src.eval.class_maps import decide
    probs = np.array([[0.5, 0.1, 0.1, 0.3]]).T          # (C, 1): clean wins without offsets
    assert decide(probs)[0] == 0
    assert decide(probs, [0.0, 0.0, 1.0])[0] == 3       # log(0.3) + 1 > log(0.5)


def test_tune_class_offsets_removes_a_rare_class_bias():
    from src.eval.class_maps import tune_class_offsets
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, 4000)
    probs = np.full((4000, 4), 0.1)
    probs[np.arange(4000), gt] = 0.4
    probs[:, 3] += 0.35                                 # model over-predicts class 3 everywhere
    probs /= probs.sum(1, keepdims=True)
    offsets, f1 = tune_class_offsets(probs, gt)
    assert offsets[2] < 0 and f1 > 0.99
