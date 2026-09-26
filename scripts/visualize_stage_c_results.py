"""
Generate the visual results for a Stage C report, mirroring
scripts/visualize_stage_b_results.py: a per-class pooled-pixel metrics bar
chart, ROC/PR curves, and a violin+box probability-by-severity plot (all
reused directly from scripts/visualize_stage_a_results.py -- they're generic
over any (N, C)-shaped input, fed here via evaluate_stage_c.py's
pool_to_grid the same way Stage B feeds them its flattened tile grid), plus
a new per-sample report figure: original / distorted / ground-truth mask /
predicted mask side by side at FULL resolution (unlike the metrics above,
this figure is for visual inspection, not tractability-limited, so it shows
the real pixel-level output the model actually produces) -- and, given a
saved training log via --log-file, the train/val loss curve.

Usage:
    python scripts/visualize_stage_c_results.py \
        --checkpoint checkpoints/stage_c/stage_c_head.pt \
        --data data/processed/stage_b --split test --device cpu \
        --log-file stage_c_<jobid>.out --out-dir docs/images
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_stage_b import flatten_tiles
from scripts.evaluate_stage_c import (
    binarize_masks,
    collect_pooled_predictions,
    load_stage_c,
    resolve_gt_threshold,
)
from scripts.visualize_stage_a_results import (
    parse_training_log,
    plot_metrics_bar_chart,
    plot_probability_by_severity,
    plot_roc_pr_curves,
    plot_training_curve,
    select_diverse_sample_indices,
)
from scripts.visualize_stage_b_results import (
    _clean_image_lookup,
    _load_image_for_row,
    class_index_for_sample,
)
from src.data.stage_c_dataset import StageCDataset
from src.eval.gate import apply_gate, collect_gate_probs, load_gate
from src.eval.metrics import compute_metrics
from src.eval.thresholds import tune_per_class_thresholds


def select_report_rows(dataset, max_probs, class_names, per_class=3, seed=0):
    """Same idea as visualize_stage_b_results.py::select_report_rows: a
    diverse set of (idx, kind, class_idx) rows covering every label kind,
    for the per-sample report figure below. `max_probs`: (N, C) per-image
    max probability (a clean sample shows its most-confident class)."""
    indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=per_class, seed=seed)
    rows = []
    for idx in indices:
        row = dataset.rows[idx]
        kind, class_idx = class_index_for_sample(row, class_names, np.asarray(max_probs[idx])[:, None, None])
        rows.append((idx, kind, class_idx))
    return rows


@torch.no_grad()
def full_resolution_predictions(backbone, head, dataset, indices, device):
    """{idx: (C, H, W) probability map} and {idx: (C, H, W) mask} for just
    the report's handful of samples."""
    probs, masks = {}, {}
    for idx in indices:
        image, mask = dataset[idx]
        probs[idx] = torch.sigmoid(head(backbone(image[None].to(device))))[0].cpu().numpy()
        masks[idx] = mask.numpy()
    return probs, masks


def plot_stage_c_report(dataset, report_rows, probs, masks, class_names, out_dir,
                         tag="", rows_per_page=6, gate_probs=None, gated_probs=None, gate_threshold=0.5):
    """Stage C per-sample report: one row per (idx, kind, class_idx) entry
    in `report_rows`, at full (native training) resolution -- no pooling,
    unlike the scalar metrics this script also writes (see pool_to_grid's
    docstring for why those pool). Columns:
      1. Original (clean) image for that sample's source image.
      2. Distorted image -- the actual dataset[idx] model input.
      3. Ground truth mask for the row's dominant class (continuous [0,1],
         viridis, 0-1 colorbar -- not thresholded, see
         src/soiling/dataset_builder.py's save_pixel_masks docstring for
         why the stored ground truth is kept continuous).
      4. Predicted mask (raw sigmoid probability) for the same class, same
         colormap/colorbar.
      5. (only when `gated_probs` is given) the same class's mask AFTER
         src.eval.gate.apply_gate -- direct comparison against column 4,
         same idea as Stage B's 8-column report's column 8, kept simpler
         here (no separate per-class raw columns; Stage C wasn't asked for
         that follow-up).

    Paginated at `rows_per_page` rows per file:
    out_dir/stage_c_full_report{tag}_page{N}.jpg. Returns the list of
    written paths. `probs`/`masks`/`gated_probs` are indexed as
    `x[idx][class_idx]` -- full (N, C, H, W) arrays or {idx: (C, H, W)} dicts."""
    n_cols = 5 if gated_probs is not None else 4
    clean_lookup = _clean_image_lookup(dataset, class_names)

    out_dir = Path(out_dir)
    pages = [report_rows[i:i + rows_per_page] for i in range(0, len(report_rows), rows_per_page)]
    if not pages:
        pages = [[]]

    out_paths = []
    for page_num, page_rows in enumerate(pages, start=1):
        n_rows = max(len(page_rows), 1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.2 * n_rows), squeeze=False)

        for r, (idx, kind, class_idx) in enumerate(page_rows):
            row = dataset.rows[idx]
            class_name = class_names[class_idx]

            image_tensor, _ = dataset[idx]
            distorted = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            clean_row = clean_lookup.get(row["source_id"])
            if clean_row is not None:
                clean = _load_image_for_row(dataset, clean_row)
                clean_title = "original"
            else:
                clean = distorted
                clean_title = "original (no clean ref. in split)"

            axes[r, 0].imshow(clean)
            axes[r, 0].set_title(clean_title, fontsize=9)
            axes[r, 0].axis("off")

            distorted_title = f"distorted ({kind})"
            if gate_probs is not None and idx in gate_probs:
                p = gate_probs[idx]
                verdict = "impaired" if p >= gate_threshold else "not impaired"
                distorted_title += f"\n[gate: {verdict} p={p:.2f}]"
            axes[r, 1].imshow(distorted)
            axes[r, 1].set_title(distorted_title, fontsize=9)
            axes[r, 1].axis("off")

            gt_mask = masks[idx][class_idx].astype(np.float32)
            im_gt = axes[r, 2].imshow(gt_mask, cmap="viridis", vmin=0, vmax=1)
            axes[r, 2].set_title(f"GT mask ({class_name})", fontsize=9)
            axes[r, 2].set_xticks([])
            axes[r, 2].set_yticks([])
            fig.colorbar(im_gt, ax=axes[r, 2], fraction=0.046)

            pred_mask = probs[idx][class_idx].astype(np.float32)
            im_pred = axes[r, 3].imshow(pred_mask, cmap="viridis", vmin=0, vmax=1)
            axes[r, 3].set_title(f"predicted mask ({class_name})", fontsize=9)
            axes[r, 3].set_xticks([])
            axes[r, 3].set_yticks([])
            fig.colorbar(im_pred, ax=axes[r, 3], fraction=0.046)

            if gated_probs is not None:
                gated_mask = gated_probs[idx][class_idx].astype(np.float32)
                im_gated = axes[r, 4].imshow(gated_mask, cmap="viridis", vmin=0, vmax=1)
                axes[r, 4].set_title(f"gated mask ({class_name})", fontsize=9)
                axes[r, 4].set_xticks([])
                axes[r, 4].set_yticks([])
                fig.colorbar(im_gated, ax=axes[r, 4], fraction=0.046)

        for r in range(len(page_rows), n_rows):
            for c in range(n_cols):
                axes[r, c].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"stage_c_full_report{tag}_page{page_num}.jpg"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/stage_c/stage_c_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--img-size", type=int, default=512, help="Must be a multiple of 32 (P5 stride)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-thresholds", action="store_true",
                         help="Use per-class best-F1 thresholds tuned on the val split (same as "
                         "evaluate_stage_c.py --tune-thresholds) instead of --threshold.")
    parser.add_argument("--eval-grid", type=int, default=64,
                         help="Pool both predictions and ground truth to this grid size before "
                         "computing the metrics bar chart/ROC-PR curves (see evaluate_stage_c.py's "
                         "pool_to_grid) -- the per-sample report figure is unaffected, always full "
                         "resolution.")
    parser.add_argument("--gt-threshold", type=float, default=None,
                         help="Pooled ground-truth coverage for a positive cell (default: the dataset's "
                         "per-class Stage B tile thresholds -- see evaluate_stage_c.py).")
    parser.add_argument("--per-kind", type=int, default=3, help="Sample images per label kind for the report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--log-file", default=None, help="Saved training stdout log (e.g. stage_c_<jobid>.out) -- if given, also plots the train/val loss curve")
    parser.add_argument("--tag", default="", help="Suffix appended to every output filename")
    parser.add_argument("--rows-per-page", type=int, default=6, help="Rows per page in the per-sample report figure")
    parser.add_argument(
        "--gate-checkpoint", default=None,
        help="Optional checkpoints/impaired_gate_severity/impaired_gate_head.pt -- if given, "
        "ACTUALLY GATES every figure/metric below (see src/eval/gate.py).",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5,
        help="P(impaired) cutoff for --gate-checkpoint: below this, an image's predictions are zeroed.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    backbone, head, class_names = load_stage_c(args.checkpoint, args.weights, device)

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = StageCDataset(args.data, split="val", img_size=args.img_size)
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
        val_probs, val_masks, _ = collect_pooled_predictions(backbone, head, val_loader, device, args.eval_grid)
        val_labels = binarize_masks(val_masks, resolve_gt_threshold(args.gt_threshold, val_set), class_names)
        threshold = tune_per_class_thresholds(*flatten_tiles(val_labels, val_probs), class_names)
        print(f"tuned thresholds: {threshold}")

    dataset = StageCDataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    probs_pooled, masks_pooled, raw_max_probs = collect_pooled_predictions(
        backbone, head, loader, device, args.eval_grid)
    max_probs = raw_max_probs

    gate_probs = None
    if args.gate_checkpoint:
        gate_head, gate_img_size = load_gate(args.gate_checkpoint, backbone.out_channels, device)
        gate_probs = collect_gate_probs(backbone, gate_head, loader, device, img_size=gate_img_size)
        probs_pooled = apply_gate(probs_pooled, gate_probs, threshold=args.gate_threshold)
        max_probs = apply_gate(raw_max_probs, gate_probs, threshold=args.gate_threshold)

    labels_pooled = binarize_masks(masks_pooled, resolve_gt_threshold(args.gt_threshold, dataset), class_names)
    labels_flat, probs_flat = flatten_tiles(labels_pooled, probs_pooled)
    metric_rows = compute_metrics(labels_flat, probs_flat, class_names, threshold=threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"stage_c_test_metrics{args.tag}.jpg"
    plot_metrics_bar_chart(metric_rows, metrics_path, title=f"Stage C pooled-pixel metrics per class{args.tag}")
    print(f"wrote {metrics_path}")

    curves_path = out_dir / f"stage_c_roc_pr_curves{args.tag}.jpg"
    plot_roc_pr_curves(labels_flat, probs_flat, class_names, curves_path, title=f"Stage C{args.tag}")
    print(f"wrote {curves_path}")

    severity_path = out_dir / f"stage_c_probability_by_severity{args.tag}.jpg"
    plot_probability_by_severity(dataset.rows, max_probs, class_names, severity_path,
                                  title=f"Stage C{args.tag} (max prob per image)")
    print(f"wrote {severity_path}")

    report_rows = select_report_rows(dataset, raw_max_probs, class_names, per_class=args.per_kind, seed=args.seed)
    report_probs, report_masks = full_resolution_predictions(
        backbone, head, dataset, [idx for idx, _, _ in report_rows], device)
    gated_report_probs = None
    if gate_probs is not None:
        gated_report_probs = {idx: p * float(gate_probs[idx] >= args.gate_threshold)
                              for idx, p in report_probs.items()}
    report_paths = plot_stage_c_report(
        dataset, report_rows, report_probs, report_masks, class_names, out_dir, tag=args.tag,
        rows_per_page=args.rows_per_page, gate_probs=gate_probs,
        gated_probs=gated_report_probs, gate_threshold=args.gate_threshold,
    )
    for p in report_paths:
        print(f"wrote {p}")

    if args.log_file:
        records = parse_training_log(Path(args.log_file).read_text())
        if not records:
            print(f"warning: no 'epoch N/M train_loss=... val_loss=...' lines found in {args.log_file}")
        else:
            curve_path = out_dir / f"stage_c_training_curve{args.tag}.jpg"
            plot_training_curve(
                records, curve_path,
                title=f"Stage C training curve{args.tag} (real TinyGPU run)",
                ylabel="loss",
            )
            print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
