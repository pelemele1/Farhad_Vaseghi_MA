"""
torch Dataset over the Stage B tile-grid dataset built by
src/soiling/dataset_builder.py::build_stage_b_dataset (a `metadata.csv` +
`images/` + `tile_labels.npy` + `stage_b_meta.json`). Image preprocessing
matches StageADataset (src/data/stage_a_dataset.py) exactly, so the same
frozen backbone sees the input format it was trained on either way.
"""
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class StageBDataset(Dataset):
    def __init__(self, data_dir, split, img_size=None, max_samples=None):
        self.data_dir = Path(data_dir)

        with open(self.data_dir / "stage_b_meta.json") as f:
            self.meta = json.load(f)

        # img_size must match what tile_labels.npy was rasterized at
        # (build_stage_b_dataset's grid = img_size // 32) -- the frozen
        # backbone's P5 output spatial size tracks whatever image size goes
        # in, so training at a different img_size than the dataset was built
        # with would silently feed the head a differently-shaped feature map
        # than the tile labels' grid, either crashing in the loss or (worse,
        # if the sizes happened to still divide evenly) misaligning tiles.
        if img_size is not None and img_size != self.meta["img_size"]:
            raise ValueError(
                f"img_size={img_size} does not match the img_size this dataset "
                f"was built with ({self.meta['img_size']}) -- the tile-label grid "
                f"({self.meta['grid_h']}x{self.meta['grid_w']}) is fixed to the "
                "build-time img_size, so training at a different size would "
                "misalign the backbone's feature-map grid against the labels. "
                "Rebuild the dataset with --img-size matching training, or omit "
                "img_size here to use the dataset's own."
            )
        self.img_size = self.meta["img_size"]
        self.class_names = tuple(self.meta["class_names"])

        with open(self.data_dir / "metadata.csv", newline="") as f:
            all_rows = list(csv.DictReader(f))
        tile_labels = np.load(self.data_dir / "tile_labels.npy")
        assert len(all_rows) == len(tile_labels), (
            f"metadata.csv has {len(all_rows)} rows but tile_labels.npy has "
            f"{len(tile_labels)} -- expected them to be row-aligned"
        )

        keep = [i for i, r in enumerate(all_rows) if r["split"] == split]
        if not keep:
            raise ValueError(
                f"no rows for split={split!r} in {self.data_dir / 'metadata.csv'}"
            )
        if max_samples is not None:
            keep = keep[:max_samples]

        self.rows = [all_rows[i] for i in keep]
        self.tile_labels = tile_labels[keep]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv.imread(str(self.data_dir / row["path"]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(self.tile_labels[idx]).float()
        return tensor, labels
