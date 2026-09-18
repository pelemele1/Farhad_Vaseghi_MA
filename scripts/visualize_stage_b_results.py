"""
Generate the visual results for a Stage B report, mirroring
scripts/visualize_stage_a_results.py: a per-class tile metrics bar chart
(reusing that script's plot_metrics_bar_chart directly -- it's generic over
any list of {class, precision, recall, f1, ap} rows), a grid of real
predictions on test-split images showing the *coarse localization*
architecture.md's own example describes ("dirt in the top right") -- each
sample's ground-truth tile grid (green) overlaid together with the model's
predicted per-tile probability (red) on the same image -- and, given a saved
training log via --log-file, the train/val loss curve (Session 18: reuses
Stage A's parse_training_log/plot_training_curve).

Multiple loss variants (bce/focal/focal-alpha75) are compared in this
project (see docs/development_log.md Session 17-18) -- pass --tag (e.g.
"_focal") so each variant's images get distinct filenames instead of
overwriting each other.

Usage:
    python scripts/visualize_stage_b_results.py \
        --checkpoint checkpoints/stage_b/stage_b_head_focal.pt \
        --data data/processed/stage_b --split test --device cpu \
        --log-file stage_b_1799134_focal.out --tag _focal --out-dir docs/images
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
from scripts.visualize_stage_a_results import (
    parse_training_log,
    plot_metrics_bar_chart,
    plot_roc_pr_curves,
    plot_training_curve,
    select_diverse_sample_indices,
)
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


def find_combo_sample_index(rows, class_names):
    """First row (in dataset order) with 2+ active classes at once, or None
    if the dataset has no combo variants (e.g. the pre-Session-19 datasets).
    Returns (idx, active_class_names)."""
    for idx, row in enumerate(rows):
        active = [c for c in class_names if int(row[c])]
        if len(active) >= 2:
            return idx, active
    return None, []


def plot_combo_sample(dataset, idx, active_classes, probs, labels, class_names, threshold, out_path):
    """One subplot per active class, each in the same GT-green/pred-red
    style as plot_tile_grid_overlay -- direct visual proof that a combo
    variant's tile grid genuinely has more than one class positive on the
    same image (Session 19+, supervisor item 5), which a single-class
    overlay (plot_tile_grid_overlay picks only the first active class)
    doesn't show."""
    image_tensor, _ = dataset[idx]
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    H, W = image.shape[:2]

    n = len(active_classes)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))
    axes = np.atleast_1d(axes).flatten()

    for ax, cls in zip(axes, active_classes):
        class_idx = class_names.index(cls)
        gt_grid = labels[idx, class_idx].astype(np.float32)
        pred_grid = probs[idx, class_idx].astype(np.float32)
        gt_up = cv.resize(gt_grid, (W, H), interpolation=cv.INTER_NEAREST)
        pred_up = cv.resize(pred_grid, (W, H), interpolation=cv.INTER_NEAREST)

        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 1] = gt_up
        overlay[..., 0] = pred_up
        overlay[..., 3] = np.clip(np.maximum(gt_up, pred_up) * 0.55, 0, 0.55)

        ax.imshow(image)
        ax.imshow(overlay)
        ax.axis("off")

        pred_active = bool(pred_grid.max() >= threshold)
        ax.set_title(
            f"{cls} (GT green / pred red)\nmax pred prob={pred_grid.max():.2f}"
            f"{'  (missed)' if not pred_active else ''}",
            fontsize=9, color=("seagreen" if pred_active else "crimson"),
        )

    fig.suptitle(f"Combo variant: {' + '.join(active_classes)} all active on the same image", fontsize=11)
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
    parser.add_argument("--log-file", default=None, help="Saved training stdout log (e.g. stage_b_<jobid>.out) -- if given, also plots the train/val loss curve")
    parser.add_argument("--tag", default="", help="Suffix (e.g. '_focal') appended to every output filename, so multiple loss variants don't overwrite each other's images")
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

    metrics_path = out_dir / f"stage_b_test_metrics{args.tag}.jpg"
    plot_metrics_bar_chart(metric_rows, metrics_path, title=f"Stage B per-tile metrics per class{args.tag}")
    print(f"wrote {metrics_path}")

    curves_path = out_dir / f"stage_b_roc_pr_curves{args.tag}.jpg"
    plot_roc_pr_curves(labels_flat, probs_flat, class_names, curves_path, title=f"Stage B{args.tag}")
    print(f"wrote {curves_path}")

    sample_indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=args.per_kind, seed=args.seed)
    grid_path = out_dir / f"stage_b_sample_predictions{args.tag}.jpg"
    plot_tile_grid_overlay(dataset, sample_indices, probs, labels, class_names, args.threshold, grid_path)
    print(f"wrote {grid_path}")

    combo_idx, combo_classes = find_combo_sample_index(dataset.rows, class_names)
    if combo_idx is not None:
        combo_path = out_dir / f"stage_b_combo_sample{args.tag}.jpg"
        plot_combo_sample(dataset, combo_idx, combo_classes, probs, labels, class_names, args.threshold, combo_path)
        print(f"wrote {combo_path}")

    if args.log_file:
        records = parse_training_log(Path(args.log_file).read_text())
        curve_path = out_dir / f"stage_b_training_curve{args.tag}.jpg"
        plot_training_curve(
            records, curve_path,
            title=f"Stage B training curve{args.tag} (real TinyGPU run)",
            ylabel="loss",
        )
        print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
