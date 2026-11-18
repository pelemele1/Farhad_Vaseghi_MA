"""
A sheet to check a built dataset's ground truth by eye: for one source photo,
the distorted image and its label map side by side, one row per contamination
type (at the chosen severity) plus a two-type example.

The label map is the dataset's own `labels/<name>.png`, colored with the
pipeline's class colors -- so what the sheet shows is exactly what a model
would be trained on, not a re-derivation of it.

Usage:
    python scripts/render_label_check_sheet.py --data data/processed/visible \
        --out docs/images/dataset/label_check.jpg --severity medium
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.pipeline import CLASS_COLORS, colorize
from src.soiling.visible_labels import CLASS_NAMES, EFFECT_NAMES

LABEL_H = 26
FONT = cv.FONT_HERSHEY_SIMPLEX


def _read_label(path):
    labels = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    return labels[..., 0] if labels.ndim == 3 else labels


def _captioned(image, text):
    """The image with a caption bar above it (image stays untouched)."""
    bar = np.full((LABEL_H, image.shape[1], 3), 255, np.uint8)
    cv.putText(bar, text, (6, 18), FONT, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    return np.vstack([bar, image])


def pick_rows(rows, severity):
    """One row per type at `severity` (single-type), plus the first two-type
    row, all from the source photo that offers the most of them."""
    by_source = defaultdict(dict)
    for row in rows:
        tokens = row["kind"].split("+")
        if row["kind"] == "clean":
            by_source[row["source_id"]]["clean"] = row
        elif len(tokens) == 1 and tokens[0].endswith(f":{severity}"):
            by_source[row["source_id"]][tokens[0].split(":")[0]] = row
        elif len(tokens) == 2:
            by_source[row["source_id"]].setdefault("two types", row)
    source_id = max(by_source, key=lambda s: len(by_source[s]))
    picked = by_source[source_id]
    order = ["clean", *EFFECT_NAMES, "two types"]
    return source_id, [(name, picked[name]) for name in order if name in picked]


def build_sheet(data_dir, severity, width):
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    source_id, picked = pick_rows(rows, severity)

    tiles = []
    for name, row in picked:
        image = cv.imread(str(data_dir / row["path"]))
        labels = _read_label(data_dir / "labels" / (Path(row["path"]).stem + ".png"))
        overlay = colorize(labels, cv.cvtColor(image, cv.COLOR_BGR2RGB))
        overlay = cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
        scale = width / 2 / image.shape[1]
        size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        pair = np.hstack([cv.resize(image, size), cv.resize(overlay, size, interpolation=cv.INTER_NEAREST)])
        share = 100.0 * (labels > 0).mean()
        caption = f"{name} ({row['kind']})" if name != "clean" else "clean"
        tiles.append(_captioned(pair, f"{caption}   |   ground truth, {share:.1f}% of the image"))

    legend = np.full((LABEL_H + 8, tiles[0].shape[1], 3), 255, np.uint8)
    x = 6
    for value, name in enumerate(CLASS_NAMES):
        color = (255, 255, 255) if name == "clean" else CLASS_COLORS[name][::-1]
        cv.rectangle(legend, (x, 8), (x + 16, 24), color, -1)
        cv.rectangle(legend, (x, 8), (x + 16, 24), (90, 90, 90), 1)
        cv.putText(legend, name, (x + 22, 22), FONT, 0.45, (0, 0, 0), 1, cv.LINE_AA)
        x += 34 + 9 * len(name)
    return source_id, np.vstack([legend, *tiles])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--out", default="docs/images/dataset/label_check.jpg")
    parser.add_argument("--severity", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--width", type=int, default=1100, help="Width of the sheet in pixels")
    args = parser.parse_args()

    source_id, sheet = build_sheet(args.data, args.severity, args.width)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out), sheet, [cv.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]}) from source photo {source_id}")


if __name__ == "__main__":
    main()
