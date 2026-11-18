"""
Sample sheets of a built visible-change dataset: images of the dataset itself,
each next to its ground truth (the dataset's own label map, colored with the
pipeline's class colors). Every sample comes from a different source photo
(chosen with a fixed seed, so the sheets are reproducible):

  samples_single_types.jpg  one row per type, light / medium / heavy
  samples_two_types.jpg     each of the 15 two-type pairs once

Usage:
    python scripts/render_dataset_samples.py --data data/processed/visible \
        --out-dir docs/images/dataset --seed 0
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.data.visible_dataset import read_label_map
from src.pipeline import CLASS_COLORS, colorize
from src.soiling.dataset_builder import COMBO_KINDS
from src.soiling.visible_labels import CLASS_NAMES, EFFECT_NAMES

THUMB_W = 300
BAR_H = 22
FONT = cv.FONT_HERSHEY_SIMPLEX
LEVEL_NAMES = {"low": "light", "medium": "medium", "high": "heavy"}


def pair_panel(data, row, title):
    """Image | ground truth, side by side, with a caption bar."""
    image = cv.imread(str(data / row["path"]))
    labels = read_label_map(data / "labels" / (Path(row["path"]).stem + ".png"))
    overlay = cv.cvtColor(colorize(labels, cv.cvtColor(image, cv.COLOR_BGR2RGB)), cv.COLOR_RGB2BGR)
    size = (THUMB_W, int(round(image.shape[0] * THUMB_W / image.shape[1])))
    pair = np.hstack([cv.resize(image, size, interpolation=cv.INTER_AREA),
                      np.full((size[1], 4, 3), 255, np.uint8),
                      cv.resize(overlay, size, interpolation=cv.INTER_NEAREST)])
    share = 100.0 * (labels > 0).mean()
    bar = np.full((BAR_H, pair.shape[1], 3), 255, np.uint8)
    cv.putText(bar, f"{title}  |  {share:.1f}% labeled  |  {row['source_id']} ({row['split']})",
               (5, 15), FONT, 0.42, (0, 0, 0), 1, cv.LINE_AA)
    return np.vstack([bar, pair])


def grid(panels, columns, gap=10):
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    blank = np.full((h, w, 3), 255, np.uint8)
    rows = []
    for i in range(0, len(panels), columns):
        cells = []
        for p in panels[i:i + columns] + [blank] * (columns - len(panels[i:i + columns])):
            cell = blank.copy()
            cell[:p.shape[0], :p.shape[1]] = p
            cells.append(cell)
            cells.append(np.full((h, gap, 3), 255, np.uint8))
        rows.append(np.hstack(cells[:-1]))
        rows.append(np.full((gap, rows[-1].shape[1], 3), 255, np.uint8))
    return np.vstack(rows[:-1])


def legend(width):
    bar = np.full((34, width, 3), 255, np.uint8)
    x = 8
    for name in CLASS_NAMES:
        color = (255, 255, 255) if name == "clean" else CLASS_COLORS[name][::-1]
        cv.rectangle(bar, (x, 9), (x + 16, 25), color, -1)
        cv.rectangle(bar, (x, 9), (x + 16, 25), (90, 90, 90), 1)
        cv.putText(bar, name, (x + 22, 23), FONT, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        x += 40 + 10 * len(name)
    return bar


def pick(rows, rng, predicate, used):
    """A random row matching `predicate`, from a photo not shown yet."""
    candidates = [r for r in rows if predicate(r) and r["source_id"] not in used]
    row = rng.choice(candidates)
    used.add(row["source_id"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--out-dir", default="docs/images/dataset")
    parser.add_argument("--seed", type=int, default=0, help="Which photos are shown")
    args = parser.parse_args()

    data, out_dir = Path(args.data), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(args.seed)
    used = set()

    singles = []
    for name in EFFECT_NAMES:
        for level in ("low", "medium", "high"):
            row = pick(rows, rng, lambda r: r["kind"] == f"{name}:{level}", used)
            singles.append(pair_panel(data, row, f"{name}, {LEVEL_NAMES[level]}"))

    pairs = []
    for combo in COMBO_KINDS:
        a, b = combo.split("+")
        row = pick(rows, rng, lambda r: sorted(t.split(":")[0] for t in r["kind"].split("+")) == sorted((a, b)),
                   used)
        levels = dict(t.split(":") for t in row["kind"].split("+"))
        pairs.append(pair_panel(data, row, f"{a} {LEVEL_NAMES[levels[a]]} + {b} {LEVEL_NAMES[levels[b]]}"))

    for name, panels, columns in (("samples_single_types.jpg", singles, 3),
                                  ("samples_two_types.jpg", pairs, 3)):
        sheet = grid(panels, columns)
        sheet = np.vstack([legend(sheet.shape[1]), sheet])
        cv.imwrite(str(out_dir / name), sheet, [cv.IMWRITE_JPEG_QUALITY, 85])
        print(f"wrote {out_dir / name} ({sheet.shape[1]}x{sheet.shape[0]}, {len(panels)} images)")


if __name__ == "__main__":
    main()
