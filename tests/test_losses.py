import csv

import torch

from src.models.losses import build_stage_a_loss, compute_pos_weight
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
