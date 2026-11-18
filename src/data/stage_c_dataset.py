"""
torch Dataset over the Stage C pixel-mask ground truth (architecture.md §2
"Stage C -- Pixel-Level Segmentation"). Reuses the exact same dataset
directory Stage B builds (a `metadata.csv` + `images/` + `masks/` +
`stage_b_meta.json`, the latter with `has_pixel_masks: true`) rather than a
separate directory -- see src/soiling/dataset_builder.py::
build_stage_b_dataset's `save_pixel_masks` docstring for why. Image
preprocessing matches StageADataset/StageBDataset exactly, so the same
frozen backbone sees the input format it was trained on either way.

`masks/{name}.npy` is an (H, W, n_classes) uint8 array, one channel per class
in EFFECT_NAMES order -- NOT a color image, so it is never passed through
cv.cvtColor(..., COLOR_BGR2RGB) the way the actual photo is. Datasets built
before Session 31 have `masks/{name}.png` instead, a 3-channel PNG; PNG holds
at most 4 channels, which is why the 6 classes needed a different container.
Both are read here.
"""
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class StageCDataset(Dataset):
    def __init__(self, data_dir, split, img_size=None, max_samples=None):
        self.data_dir = Path(data_dir)

        with open(self.data_dir / "stage_b_meta.json") as f:
            self.meta = json.load(f)

        if not self.meta.get("has_pixel_masks", False):
            raise ValueError(
                f"{self.data_dir} was not built with save_pixel_masks=True (no masks/ folder) "
                "-- rebuild with --save-pixel-masks (scripts/build_stage_b_dataset.py) to use "
                "it for Stage C."
            )

        self.img_size = self.meta["img_size"] if img_size is None else img_size
        if self.img_size % 32 != 0:
            raise ValueError(
                f"img_size must be a multiple of 32 (P5 stride, so the decoder's 5x "
                f"2x-upsample stack lands back on exactly img_size), got {self.img_size}"
            )
        self.class_names = tuple(self.meta["class_names"])

        with open(self.data_dir / "metadata.csv", newline="") as f:
            all_rows = list(csv.DictReader(f))

        keep = [r for r in all_rows if r["split"] == split]
        if not keep:
            raise ValueError(
                f"no rows for split={split!r} in {self.data_dir / 'metadata.csv'}"
            )
        if max_samples is not None:
            keep = keep[:max_samples]

        self.rows = keep

    def __len__(self):
        return len(self.rows)

    def _mask_path(self, row):
        npy = self.data_dir / "masks" / (Path(row["path"]).stem + ".npy")
        return npy if npy.exists() else npy.with_suffix(".png")

    def _read_mask(self, row):
        path = self._mask_path(row)
        if path.suffix == ".npy":
            return np.load(path)
        return cv.imread(str(path), cv.IMREAD_UNCHANGED)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv.imread(str(self.data_dir / row["path"]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = self._read_mask(row)
        mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0

        return image_tensor, mask_tensor
