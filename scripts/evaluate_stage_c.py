"""
Evaluate a trained Stage C pixel-level segmentation head (architecture.md
§2): per-class precision/recall/F1/AP/ROC-AUC over individual *pixels*
(pooled to a tractable grid, not one row per raw pixel -- see
`pool_to_grid`'s docstring for why) on a held-out split of the dataset built
by src/soiling/dataset_builder.py::build_stage_b_dataset(save_pixel_masks=
True).

Usage:
    python scripts/evaluate_stage_c.py --checkpoint checkpoints/stage_c/stage_c_head.pt \
        --data data/processed/stage_b_scratch15 --split test --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_stage_b import _print_metrics_table, flatten_tiles
from src.data.stage_c_dataset import StageCDataset
from src.eval.gate import apply_gate, collect_gate_probs
from src.eval.metrics import collect_predictions, compute_metrics
from src.eval.severity import broadcast_rows_to_tiles, compute_metrics_by_severity
from src.eval.thresholds import tune_per_class_thresholds
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import ImpairedGateHead, StageCDistortionHead


def pool_to_grid(array, grid_size):
    """array: (N, C, H, W) float array (a predicted probability map or a
    continuous [0,1] ground-truth mask), at Stage C's native (e.g. 512x512)
    resolution. Area-average-pools each sample+class down to (grid_size,
    grid_size) via cv.resize(INTER_AREA) -- the exact same downsampling
    src/soiling/tile_labels.py::rasterize_tile_label already uses to turn a
    full-resolution mask into a coarse coverage grid, reused here purely for
    computational tractability: the sklearn-based ranking metrics in
    compute_metrics/tune_per_class_thresholds sort every row, which is fine
    at Stage B's 16x16-tile scale (a few hundred thousand rows for a whole
    split) but would be ~367 million rows at Stage C's native 512x512 --
    far too slow/memory-heavy for those sort-based implementations. Pooling
    to a smaller (default 64x64) grid at EVALUATION time only keeps the
    exact same metric definitions used everywhere else in this project
    (via flatten_tiles + compute_metrics, unmodified) tractable, without
    writing a new metric implementation. Training is unaffected -- the loss
    still supervises the full native resolution; only this scalar-metrics
    table is computed on the pooled grid."""
    n, c, h, w = array.shape
    pooled = np.empty((n, c, grid_size, grid_size), dtype=np.float32)
    for i in range(n):
        for j in range(c):
            pooled[i, j] = cv.resize(
                array[i, j].astype(np.float32), (grid_size, grid_size), interpolation=cv.INTER_AREA
            )
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage_c/stage_c_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b_scratch15")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--img-size", type=int, default=512, help="Must be a multiple of 32 (P5 stride)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-grid", type=int, default=64,
                         help="Pool both predictions and ground truth to this grid size before "
                         "computing metrics (see pool_to_grid) -- default 64 is 16x finer than "
                         "Stage B's 16x16 tile grid while staying tractable for sklearn's "
                         "sort-based AP/ROC-AUC.")
    parser.add_argument("--gt-threshold", type=float, default=0.5,
                         help="Cutoff on the pooled [0,1] ground-truth mask value for a cell to "
                         "count as a genuine positive -- P/R/F1/AP/ROC-AUC need binary ground "
                         "truth; the stored mask is kept continuous (see StageCDataset), so this "
                         "binarizes it at evaluation time instead of at dataset-build time, the "
                         "same 'decide the threshold at eval time' convention this project "
                         "already follows for the *predicted*-probability decision threshold "
                         "below.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for precision/recall/F1")
    parser.add_argument(
        "--tune-thresholds", action="store_true",
        help="Tune a per-class best-F1 threshold on the val split, then evaluate --split with those "
        "instead of the single --threshold value for every class (see src/eval/thresholds.py).",
    )
    parser.add_argument(
        "--gate-checkpoint", default=None,
        help="Optional checkpoints/impaired_gate_severity/impaired_gate_head.pt -- if given, ALSO "
        "prints a second, gated evaluation table where Stage C's pixel predictions are zeroed for "
        "any image the gate calls 'not impaired' (see src/eval/gate.py).",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5,
        help="P(impaired) cutoff for --gate-checkpoint: below this, an image's predictions are zeroed.",
    )
    parser.add_argument(
        "--by-severity", action="store_true",
        help="Also print a per-class, per-severity-level (low/medium/high) pooled-pixel-metrics "
        "breakdown (requires a dataset built with --include-severity; see src/eval/severity.py).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageCDistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = StageCDataset(args.data, split="val", img_size=args.img_size)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        val_probs, val_masks = collect_predictions(backbone, head, val_loader, device)
        val_probs_pooled = pool_to_grid(val_probs, args.eval_grid)
        val_labels_pooled = (pool_to_grid(val_masks, args.eval_grid) >= args.gt_threshold).astype(np.uint8)
        val_labels_flat, val_probs_flat = flatten_tiles(val_labels_pooled, val_probs_pooled)
        threshold = tune_per_class_thresholds(val_labels_flat, val_probs_flat, class_names)
        print("Tuned per-class thresholds (best-F1 on val split):")
        for name in class_names:
            print(f"  {name:<10}{threshold[name]:.3f}")

    dataset = StageCDataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs, masks = collect_predictions(backbone, head, loader, device)
    probs_pooled = pool_to_grid(probs, args.eval_grid)
    labels_pooled = (pool_to_grid(masks, args.eval_grid) >= args.gt_threshold).astype(np.uint8)
    labels_flat, probs_flat = flatten_tiles(labels_pooled, probs_pooled)
    rows = compute_metrics(labels_flat, probs_flat, class_names, threshold=threshold)

    n_cells = labels_flat.shape[0]
    threshold_desc = "tuned per-class" if args.tune_thresholds else args.threshold
    _print_metrics_table(rows, len(dataset), n_cells, args.split, threshold_desc,
                          label="ungated" if args.gate_checkpoint else None)

    if args.gate_checkpoint:
        gate_ckpt = torch.load(args.gate_checkpoint, map_location=device, weights_only=False)
        gate_head = ImpairedGateHead(in_channels=backbone.out_channels).to(device)
        gate_head.load_state_dict(gate_ckpt["head_state_dict"])
        gate_head.eval()
        gate_probs = collect_gate_probs(backbone, gate_head, loader, device)
        gated_probs = apply_gate(probs, gate_probs, threshold=args.gate_threshold)

        gated_probs_pooled = pool_to_grid(gated_probs, args.eval_grid)
        gated_labels_flat, gated_probs_flat = flatten_tiles(labels_pooled, gated_probs_pooled)
        gated_rows = compute_metrics(gated_labels_flat, gated_probs_flat, class_names, threshold=threshold)
        print()
        _print_metrics_table(gated_rows, len(dataset), n_cells, args.split, threshold_desc, label="gated")

    if args.by_severity:
        broadcast_rows = broadcast_rows_to_tiles(dataset.rows, args.eval_grid, args.eval_grid)
        by_class = compute_metrics_by_severity(broadcast_rows, labels_flat, probs_flat, class_names, threshold)
        print(f"\nPer-severity breakdown (ungated, pooled to {args.eval_grid}x{args.eval_grid}):")
        print(f"{'class':<10}{'severity':<10}{'threshold':>10}{'precision':>10}{'recall':>10}{'f1':>10}{'AP':>10}{'ROC-AUC':>10}{'support':>10}")
        for name in class_names:
            for level, row in by_class[name].items():
                print(f"{name:<10}{level:<10}{row['threshold']:>10.3f}{row['precision']:>10.3f}{row['recall']:>10.3f}"
                      f"{row['f1']:>10.3f}{row['ap']:>10.3f}{row['roc_auc']:>10.3f}{row['support']:>10}")


if __name__ == "__main__":
    main()
