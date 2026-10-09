"""
torch Dataset over the visible-change dataset (src/soiling/visible_dataset.py),
serving the same images to all three stages with the target each one needs:
  "image": float (3,) -- which distortions are visible anywhere (Stage A)
  "tile":  long (grid, grid) -- class per tile, 0 clean .. 3 scratch (Stage B)
  "pixel": long (img_size, img_size) -- class per pixel (Stage C)
Images are preprocessed like the other datasets (BGR -> RGB, resize, /255).
"""
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

TARGETS = ("image", "tile", "pixel")


def read_label_map(path):
    """(H, W) uint8 class map. ultralytics (imported with the backbone) patches
    cv2.imread to return single-channel images as (H, W, 1)."""
    labels = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    return labels[..., 0] if labels.ndim == 3 else labels


class VisibleDataset(Dataset):
    def __init__(self, data_dir, split, target="pixel", img_size=None, max_samples=None, hflip=False):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        self.data_dir = Path(data_dir)
        self.target = target
        self.hflip = hflip
        with open(self.data_dir / "visible_meta.json") as f:
            self.meta = json.load(f)
        self.img_size = img_size or self.meta["img_size"]
        if target == "tile" and self.img_size != self.meta["img_size"]:
            raise ValueError(f"tile labels were built for img_size={self.meta['img_size']}, got {self.img_size}")
        self.class_names = tuple(self.meta["class_names"])            # clean, dirt, water, scratch
        self.effect_names = self.class_names[1:]

        with open(self.data_dir / "metadata.csv", newline="") as f:
            all_rows = list(csv.DictReader(f))
        keep = [i for i, r in enumerate(all_rows) if r["split"] == split]
        if not keep:
            raise ValueError(f"no rows for split={split!r} in {self.data_dir / 'metadata.csv'}")
        if max_samples is not None:
            keep = keep[:max_samples]
        self.rows = [all_rows[i] for i in keep]
        if target == "tile":
            self.tile_labels = np.load(self.data_dir / "tile_labels.npy")[keep]

    def __len__(self):
        return len(self.rows)

    def label_path(self, row):
        return self.data_dir / "labels" / (Path(row["path"]).stem + ".png")

    def load_label_map(self, idx, size=None):
        """(size, size) uint8 class map, nearest-neighbour resized."""
        size = size or self.img_size
        labels = read_label_map(self.label_path(self.rows[idx]))
        return cv.resize(labels, (size, size), interpolation=cv.INTER_NEAREST)

    def image_labels(self, idx):
        return np.array([float(self.rows[idx][n]) for n in self.effect_names], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv.imread(str(self.data_dir / row["path"]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.target == "image":
            target = torch.from_numpy(self.image_labels(idx))
        elif self.target == "tile":
            target = torch.from_numpy(self.tile_labels[idx].astype(np.int64))
        else:
            target = torch.from_numpy(self.load_label_map(idx).astype(np.int64))

        if self.hflip and torch.rand(1).item() < 0.5:
            tensor = tensor.flip(-1)
            if self.target != "image":
                target = target.flip(-1)
        return tensor, target
