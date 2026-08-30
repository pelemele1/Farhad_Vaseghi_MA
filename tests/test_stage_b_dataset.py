import csv
import json

import cv2 as cv
import numpy as np
import pytest
import torch

from src.data.stage_b_dataset import StageBDataset
from src.soiling.dataset_builder import EFFECT_NAMES


def _write_fake_stage_b_dataset(data_dir, rows_spec, grid=(2, 2), img_size=64):
    """rows_spec: list of (name, split, dirt, water, scratch, tile_grid or
    None). tile_grid, if given, is a (3, grid_h, grid_w) array; otherwise an
    all-zero grid is used."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    tile_labels = []
    for name, split, dirt, water, scratch, tile_grid in rows_spec:
        img = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
        cv.imwrite(str(images_dir / f"{name}.jpg"), img)
        rows.append({
            "path": f"images/{name}.jpg", "source_id": name, "variant_id": 0,
            "split": split, "dirt": dirt, "water": water, "scratch": scratch,
        })
        if tile_grid is None:
            tile_grid = np.zeros((3, *grid), dtype=np.uint8)
        tile_labels.append(tile_grid)

    with open(data_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_id", "variant_id", "split", *EFFECT_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    np.save(data_dir / "tile_labels.npy", np.stack(tile_labels).astype(np.uint8))
    with open(data_dir / "stage_b_meta.json", "w") as f:
        json.dump({
            "img_size": img_size, "grid_h": grid[0], "grid_w": grid[1],
            "class_names": list(EFFECT_NAMES), "thresholds": {},
        }, f)

    return rows


def test_len_and_split_filtering(tmp_path):
    _write_fake_stage_b_dataset(tmp_path, [
        ("a", "train", 1, 0, 0, None),
        ("b", "train", 0, 1, 0, None),
        ("c", "val", 0, 0, 1, None),
    ])
    train_set = StageBDataset(tmp_path, split="train")
    val_set = StageBDataset(tmp_path, split="val")
    assert len(train_set) == 2
    assert len(val_set) == 1


def test_getitem_shape_dtype_and_tile_labels(tmp_path):
    grid = np.zeros((3, 2, 2), dtype=np.uint8)
    grid[0, 0, 1] = 1  # dirt positive in the top-right tile
    _write_fake_stage_b_dataset(tmp_path, [("a", "train", 1, 0, 0, grid)])

    dataset = StageBDataset(tmp_path, split="train")
    image, labels = dataset[0]

    assert image.shape == (3, 64, 64)  # from stage_b_meta.json's img_size
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0

    assert labels.shape == (3, 2, 2)
    assert labels.dtype == torch.float32
    assert torch.equal(labels, torch.from_numpy(grid).float())


def test_img_size_matching_meta_is_accepted(tmp_path):
    _write_fake_stage_b_dataset(tmp_path, [("a", "train", 0, 0, 0, None)], img_size=64)
    dataset = StageBDataset(tmp_path, split="train", img_size=64)
    image, _ = dataset[0]
    assert image.shape == (3, 64, 64)


def test_img_size_mismatch_with_meta_raises(tmp_path):
    # A different img_size than the dataset was built at would misalign the
    # backbone's feature-map grid against tile_labels.npy's fixed grid --
    # must fail loudly, not silently resize.
    _write_fake_stage_b_dataset(tmp_path, [("a", "train", 0, 0, 0, None)], img_size=64)
    with pytest.raises(ValueError):
        StageBDataset(tmp_path, split="train", img_size=32)


def test_class_names_loaded_from_meta(tmp_path):
    _write_fake_stage_b_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    dataset = StageBDataset(tmp_path, split="train")
    assert dataset.class_names == EFFECT_NAMES


def test_max_samples_truncates(tmp_path):
    _write_fake_stage_b_dataset(tmp_path, [
        ("a", "train", 0, 0, 0, None),
        ("b", "train", 0, 0, 0, None),
        ("c", "train", 0, 0, 0, None),
    ])
    dataset = StageBDataset(tmp_path, split="train", max_samples=2)
    assert len(dataset) == 2


def test_missing_split_raises(tmp_path):
    _write_fake_stage_b_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    with pytest.raises(ValueError):
        StageBDataset(tmp_path, split="test")
