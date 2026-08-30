"""
Generate the visual results for a Stage B report, mirroring
scripts/visualize_stage_a_results.py: a per-class tile metrics bar chart
(reusing that script's plot_metrics_bar_chart directly -- it's generic over
any list of {class, precision, recall, f1, ap} rows), and a grid of real
predictions on test-split images showing the *coarse localization*
architecture.md's own example describes ("dirt in the top right") -- each
sample's ground-truth tile grid (green) overlaid together with the model's
predicted per-tile probability (red) on the same image.

Usage:
    python scripts/visualize_stage_b_results.py \
        --checkpoint checkpoints/stage_b/stage_b_head.pt \
        --data data/processed/stage_b --split test --device cpu \
        --out-dir docs/images
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_stage_b import flatten_tiles
from scripts.visualize_stage_a_results import plot_metrics_bar_chart, select_diverse_sample_indices
from src.data.stage_b_dataset import StageBDataset
from src.eval.metrics import collect_predictions, compute_metrics
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead


def class_index_for_sample(row, class_names, probs_chw=None):
    """row: a metadata.csv row dict (string 0/1 values for each class name).
    Returns (kind, class_idx): `kind` is the sample's active class name, or
    "clean" if none; `class_idx` is which class's tile grid to visualize --
    the active class itself, or (for a clean sample, given `probs_chw`, a
    (C, H, W) prediction array) whichever class the model was most
    confident about anywhere in the grid -- a direct check for false
    positives on undistorted images. Pulled out of plot_tile_grid_overlay so
    the sample-picking logic is unit-testable without matplotlib."""
    active = [c for c in class_names if int(row[c])]
    if active:
        kind = active[0]
        return kind, class_names.index(kind)
    class_idx = int(np.argmax(probs_chw.max(axis=(1, 2)))) if probs_chw is not None else 0
    return "clean", class_idx


def plot_tile_grid_overlay(dataset, indices, probs, labels, class_names, threshold, out_path, cols=3):
    """For each selected sample: the ground-truth tile grid for its active
    class (green) and the model's predicted per-tile probability for that
    same class (red), both upsampled to image resolution with nearest-
    neighbor interpolation so tile boundaries stay crisp, overlaid together
    on the actual image. A "clean" sample (no active class) instead shows
    whichever class the model was most (potentially wrongly) confident
    about -- a direct check for false positives on undistorted images."""
    n = len(indices)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 4.6 * rows_n))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        image_tensor, _ = dataset[idx]
        image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        H, W = image.shape[:2]

        row = dataset.rows[idx]
        kind, class_idx = class_index_for_sample(row, class_names, probs[idx])
        active = kind != "clean"

        gt_grid = labels[idx, class_idx].astype(np.float32)
        pred_grid = probs[idx, class_idx].astype(np.float32)
        gt_up = cv.resize(gt_grid, (W, H), interpolation=cv.INTER_NEAREST)
        pred_up = cv.resize(pred_grid, (W, H), interpolation=cv.INTER_NEAREST)

        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 1] = gt_up  # green channel = ground truth
        overlay[..., 0] = pred_up  # red channel = predicted probability
        overlay[..., 3] = np.clip(np.maximum(gt_up, pred_up) * 0.55, 0, 0.55)

        ax.imshow(image)
        ax.imshow(overlay)
        ax.axis("off")

        pred_active = bool(pred_grid.max() >= threshold)
        correct = pred_active == bool(active)
        ax.set_title(
            f"class={kind} ({class_names[class_idx]} shown)\n"
            f"GT green / pred red, max pred prob={pred_grid.max():.2f}",
            fontsize=9, color=("seagreen" if correct else "crimson"),
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/stage_b/stage_b_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--per-kind", type=int, default=3, help="Sample images per label kind for the grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="docs/images")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageBDistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    dataset = StageBDataset(args.data, split=args.split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs, labels = collect_predictions(backbone, head, loader, device)

    labels_flat, probs_flat = flatten_tiles(labels, probs)
    metric_rows = compute_metrics(labels_flat, probs_flat, class_names, threshold=args.threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "stage_b_test_metrics.jpg"
    plot_metrics_bar_chart(metric_rows, metrics_path, title="Stage B per-tile metrics per class")
    print(f"wrote {metrics_path}")

    sample_indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=args.per_kind, seed=args.seed)
    grid_path = out_dir / "stage_b_sample_predictions.jpg"
    plot_tile_grid_overlay(dataset, sample_indices, probs, labels, class_names, args.threshold, grid_path)
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
