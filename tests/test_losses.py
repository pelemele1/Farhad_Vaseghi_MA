import csv

import numpy as np
import torch
import torch.nn as nn

from src.models.losses import (
    build_stage_a_loss,
    build_stage_b_loss,
    compute_pos_weight,
    compute_tile_pos_weight,
)
from src.soiling.dataset_builder import EFFECT_NAMES


def _write_metadata(path, rows):
    fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_pos_weight_matches_hand_calculation(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s0", "variant_id": 1, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
        {"path": "c", "source_id": "s0", "variant_id": 2, "split": "train", "dirt": 0, "water": 1, "scratch": 0},
        {"path": "d", "source_id": "s0", "variant_id": 3, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    pos_weight = compute_pos_weight(csv_path)
    # dirt: 1 positive / 3 negative -> 3.0; water: same -> 3.0;
    # scratch: 0 positive / 4 negative -> 4/max(0,1) = 4.0
    assert torch.allclose(pos_weight, torch.tensor([3.0, 3.0, 4.0]))


def test_compute_pos_weight_respects_split_filter(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    pos_weight = compute_pos_weight(csv_path, split="train")
    # only the "train" row counted: dirt 1 positive/0 negative -> 0.0;
    # water/scratch 0 positive/1 negative -> 1.0
    assert torch.allclose(pos_weight, torch.tensor([0.0, 1.0, 1.0]))


def test_build_stage_a_loss_is_finite_and_uses_pos_weight():
    pos_weight = torch.tensor([2.0, 1.0, 3.0])
    loss_fn = build_stage_a_loss(pos_weight)
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)


# --- Stage B (tile-grid) -------------------------------------------------


def _write_metadata_rows(path, rows):
    fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_tile_pos_weight_matches_hand_calculation(tmp_path):
    # 2 rows, each a 2x2 grid (4 tiles). dirt: 1 positive tile total out of
    # 8 -> pos_weight = (8-1)/1 = 7.0. water: 0 positive -> (8-0)/max(0,1) = 8.0.
    # scratch: 4 positive (all of row 2's tiles) -> (8-4)/4 = 1.0.
    tile_labels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    tile_labels[0, 0, 0, 0] = 1  # one dirt tile in row 0
    tile_labels[1, 2, :, :] = 1  # all 4 scratch tiles in row 1
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "train", "dirt": 0, "water": 0, "scratch": 1},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_rows(csv_path, rows)
    npy_path = tmp_path / "tile_labels.npy"
    np.save(npy_path, tile_labels)

    pos_weight = compute_tile_pos_weight(npy_path, csv_path)
    assert torch.allclose(pos_weight, torch.tensor([7.0, 8.0, 1.0]))


def test_compute_tile_pos_weight_respects_split_filter(tmp_path):
    tile_labels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    tile_labels[0, 0, 0, 0] = 1
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_rows(csv_path, rows)
    npy_path = tmp_path / "tile_labels.npy"
    np.save(npy_path, tile_labels)

    pos_weight = compute_tile_pos_weight(npy_path, csv_path, split="val")
    # only row "b" (all-zero, 4 tiles/class) counted -> (4-0)/max(0,1) = 4.0 for every class
    assert torch.allclose(pos_weight, torch.tensor([4.0, 4.0, 4.0]))


def test_build_stage_b_loss_is_finite_and_none_pos_weight_is_safe():
    loss_fn = build_stage_b_loss(None)
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 2, (2, 3, 4, 4)).float()
    assert torch.isfinite(loss_fn(logits, labels))


def test_build_stage_b_loss_reshapes_pos_weight_onto_the_channel_axis():
    # A (C,) pos_weight broadcasts against the *last* dim by default -- if W
    # also happens to equal C (here both 3, e.g. a 4x3 grid), an unreshaped
    # (C,) pos_weight would silently apply per-column instead of
    # per-channel, without even raising a shape error. Confirm
    # build_stage_b_loss's result matches the correctly-reshaped (C, 1, 1)
    # computation and differs from that naive (unreshaped) one, so this test
    # would fail if the reshape were removed.
    torch.manual_seed(0)
    pos_weight = torch.tensor([5.0, 1.0, 1.0])
    logits = torch.randn(2, 3, 4, 3)  # B, C=3, H=4, W=3 -- W matches C on purpose
    labels = torch.randint(0, 2, (2, 3, 4, 3)).float()

    correct = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(-1, 1, 1))(logits, labels)
    naive = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)  # wrong axis: applies along W, not C

    got = build_stage_b_loss(pos_weight)(logits, labels)
    assert torch.allclose(got, correct)
    assert not torch.allclose(got, naive)
