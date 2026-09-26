import csv
import json

import cv2 as cv
import numpy as np
import pytest
import torch

from src.data.stage_c_dataset import StageCDataset
from src.soiling.dataset_builder import EFFECT_NAMES


def _write_fake_stage_c_dataset(data_dir, rows_spec, img_size=64, has_pixel_masks=True,
                                 mask_size=(40, 60)):
    """rows_spec: list of (name, split, dirt, water, scratch, mask or None).
    mask, if given, is a (H, W, 3) uint8 array (dirt/water/scratch channels);
    otherwise an all-zero mask is used."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    masks_dir = data_dir / "masks"
    masks_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for name, split, dirt, water, scratch, mask in rows_spec:
        img = rng.integers(0, 255, size=(*mask_size, 3), dtype=np.uint8)
        cv.imwrite(str(images_dir / f"{name}.jpg"), img)
        if mask is None:
            mask = np.zeros((*mask_size, 3), dtype=np.uint8)
        cv.imwrite(str(masks_dir / f"{name}.png"), mask)
        rows.append({
            "path": f"images/{name}.jpg", "source_id": name, "variant_id": 0,
            "split": split, "dirt": dirt, "water": water, "scratch": scratch,
        })

    with open(data_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_id", "variant_id", "split", *EFFECT_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    with open(data_dir / "stage_b_meta.json", "w") as f:
        json.dump({
            "img_size": img_size, "grid_h": 2, "grid_w": 2,
            "class_names": list(EFFECT_NAMES), "thresholds": {},
            "has_pixel_masks": has_pixel_masks,
        }, f)

    return rows


def test_len_and_split_filtering(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [
        ("a", "train", 1, 0, 0, None),
        ("b", "train", 0, 1, 0, None),
        ("c", "val", 0, 0, 1, None),
    ])
    train_set = StageCDataset(tmp_path, split="train")
    val_set = StageCDataset(tmp_path, split="val")
    assert len(train_set) == 2
    assert len(val_set) == 1


def test_getitem_shape_dtype_and_mask_values(tmp_path):
    mask = np.zeros((40, 60, 3), dtype=np.uint8)
    mask[:, :, 0] = 255  # dirt channel fully "on"
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 1, 0, 0, mask)])

    dataset = StageCDataset(tmp_path, split="train")
    image, mask_tensor = dataset[0]

    assert image.shape == (3, 64, 64)  # from stage_b_meta.json's img_size
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0

    assert mask_tensor.shape == (3, 64, 64)
    assert mask_tensor.dtype == torch.float32
    assert torch.allclose(mask_tensor[0], torch.ones(64, 64), atol=1e-5)
    assert torch.allclose(mask_tensor[1], torch.zeros(64, 64), atol=1e-5)
    assert torch.allclose(mask_tensor[2], torch.zeros(64, 64), atol=1e-5)


def test_custom_img_size_resizes_both_image_and_mask(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    dataset = StageCDataset(tmp_path, split="train", img_size=128)
    image, mask_tensor = dataset[0]
    assert image.shape == (3, 128, 128)
    assert mask_tensor.shape == (3, 128, 128)


def test_img_size_not_multiple_of_32_raises(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    with pytest.raises(ValueError):
        StageCDataset(tmp_path, split="train", img_size=100)


def test_missing_pixel_masks_raises(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 0, 0, 0, None)], has_pixel_masks=False)
    with pytest.raises(ValueError):
        StageCDataset(tmp_path, split="train")


def test_class_names_loaded_from_meta(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    dataset = StageCDataset(tmp_path, split="train")
    assert dataset.class_names == EFFECT_NAMES


def test_max_samples_truncates(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [
        ("a", "train", 0, 0, 0, None),
        ("b", "train", 0, 0, 0, None),
        ("c", "train", 0, 0, 0, None),
    ])
    dataset = StageCDataset(tmp_path, split="train", max_samples=2)
    assert len(dataset) == 2


def test_missing_split_raises(tmp_path):
    _write_fake_stage_c_dataset(tmp_path, [("a", "train", 0, 0, 0, None)])
    with pytest.raises(ValueError):
        StageCDataset(tmp_path, split="test")
