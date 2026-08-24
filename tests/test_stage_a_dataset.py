import csv

import cv2 as cv
import numpy as np
import pytest
import torch

from src.data.stage_a_dataset import StageADataset
from src.soiling.dataset_builder import EFFECT_NAMES


def _write_fake_dataset(data_dir, rows_spec):
    """rows_spec: list of (name, split, dirt, water, scratch) tuples."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for name, split, dirt, water, scratch in rows_spec:
        img = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
        cv.imwrite(str(images_dir / f"{name}.jpg"), img)
        rows.append({
            "path": f"images/{name}.jpg", "source_id": name, "variant_id": 0,
            "split": split, "dirt": dirt, "water": water, "scratch": scratch,
        })
    with open(data_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_id", "variant_id", "split", *EFFECT_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_len_and_split_filtering(tmp_path):
    _write_fake_dataset(tmp_path, [
        ("a", "train", 1, 0, 0),
        ("b", "train", 0, 1, 0),
        ("c", "val", 0, 0, 1),
    ])
    train_set = StageADataset(tmp_path, split="train")
    val_set = StageADataset(tmp_path, split="val")
    assert len(train_set) == 2
    assert len(val_set) == 1


def test_getitem_shape_dtype_and_labels(tmp_path):
    _write_fake_dataset(tmp_path, [("a", "train", 1, 0, 1)])
    dataset = StageADataset(tmp_path, split="train", img_size=32)
    image, labels = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert torch.equal(labels, torch.tensor([1.0, 0.0, 1.0]))


def test_max_samples_truncates(tmp_path):
    _write_fake_dataset(tmp_path, [
        ("a", "train", 0, 0, 0),
        ("b", "train", 0, 0, 0),
        ("c", "train", 0, 0, 0),
    ])
    dataset = StageADataset(tmp_path, split="train", max_samples=2)
    assert len(dataset) == 2


def test_missing_split_raises(tmp_path):
    _write_fake_dataset(tmp_path, [("a", "train", 0, 0, 0)])
    with pytest.raises(ValueError):
        StageADataset(tmp_path, split="test")
