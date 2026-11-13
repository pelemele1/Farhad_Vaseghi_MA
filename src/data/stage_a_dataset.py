"""
torch Dataset over the Stage A distortion dataset built by
src/soiling/dataset_builder.py (a `metadata.csv` + `images/` directory).
Images are read/resized/normalized the way Ultralytics' own predictor
preprocesses input (BGR -> RGB, resize, HWC -> CHW, /255) so the frozen YOLO
backbone (src/models/backbone.py) sees the input format it was trained on.
"""
import csv
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import Dataset

from src.soiling.dataset_builder import EFFECT_NAMES


class StageADataset(Dataset):
    def __init__(self, data_dir, split, img_size=640, max_samples=None):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        with open(self.data_dir / "metadata.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not rows:
            raise ValueError(
                f"no rows for split={split!r} in {self.data_dir / 'metadata.csv'}"
            )
        self.rows = rows[:max_samples] if max_samples is not None else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv.imread(str(self.data_dir / row["path"]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels = torch.tensor([float(row[name]) for name in EFFECT_NAMES])
        return tensor, labels


class ImpairedGateDataset(StageADataset):
    """Same images/preprocessing as StageADataset, but returns a single
    binary "is this image impaired at all" label instead of the 3-class
    multi-hot label, derived on the fly as the OR of dirt/water/scratch --
    no new metadata.csv column or dataset rebuild needed. Label dtype is
    int64 (nn.CrossEntropyLoss's required target dtype), unlike
    StageADataset's float multi-hot labels.
    """

    def __getitem__(self, idx):
        tensor, multi_label = super().__getitem__(idx)
        impaired = torch.tensor(int(multi_label.sum().item() > 0), dtype=torch.long)
        return tensor, impaired
