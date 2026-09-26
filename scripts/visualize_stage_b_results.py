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
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

from scripts.evaluate_stage_b import flatten_tiles
from scripts.visualize_stage_a_results import (
    parse_training_log,
    plot_metrics_bar_chart,
    plot_probability_by_severity,
    plot_roc_pr_curves,
    plot_training_curve,
    select_diverse_sample_indices,
)
from src.data.stage_b_dataset import StageBDataset
from src.eval.gate import apply_gate, collect_gate_probs, load_gate
from src.eval.metrics import collect_predictions, compute_metrics
from src.eval.thresholds import threshold_for, tune_per_class_thresholds
from scripts.train_stage_b import load_stage_b


def max_prob_per_image(probs):
    """(N, C, H, W) per-tile probability array -> (N, C): each image's max
    predicted probability per class, over the whole tile grid. Used to feed
    Stage B's per-tile predictions into
    visualize_stage_a_results.py::plot_probability_by_severity, which
    expects one probability per image per class (severity is an
    image-level property, not a per-tile one) -- same "max over the grid"
    aggregation scripts/diagnose_clean_false_positives.py already uses for
    its clean-image false-positive check."""
    return probs.max(axis=(2, 3))


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

        pred_active = bool(pred_grid.max() >= threshold_for(threshold, class_names[class_idx]))
        correct = pred_active == bool(active)
        ax.set_title(
            f"class={kind} ({class_names[class_idx]} shown)\n"
            f"GT green / pred red, max pred prob={pred_grid.max():.2f}",
            fontsize=9, color=("seagreen" if correct else "crimson"),
        )

    for ax in axes[n:]:
        ax.axis("off")

    # Composite-overlay style (color = blend of GT + prediction) can't be
    # legended with a single 0-1 colorbar -- point readers to the new
    # per-tile-probability figure instead of adding one here (Session 20).
    fig.text(
        0.5, 0.01,
        "Overlay color = blend of GT (green) and prediction (red); "
        "see stage_b_full_report*.jpg for per-tile probability values with a 0-1 colorbar.",
        ha="center", fontsize=8, color="gray",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
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

        pred_active = bool(pred_grid.max() >= threshold_for(threshold, cls))
        ax.set_title(
            f"{cls} (GT green / pred red)\nmax pred prob={pred_grid.max():.2f}"
            f"{'  (missed)' if not pred_active else ''}",
            fontsize=9, color=("seagreen" if pred_active else "crimson"),
        )

    fig.suptitle(f"Combo variant: {' + '.join(active_classes)} all active on the same image", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --- Stage B per-sample report figure (Session 20, supervisor item 1; --
# --- originally 5 columns, extended to 7 per a later follow-up to show   --
# --- all 3 classes' probability tiles instead of just the dominant one) --


def _compute_per_class_pr_curves(labels_flat, probs_flat, class_names):
    """One (recall, precision, ap) tuple per class, computed once over the
    WHOLE flattened test-split tile arrays -- AUC-PR is not a per-sample
    quantity, so every row of a given class in the per-sample report reuses
    this same cached curve instead of recomputing it per row. A class's
    entry is None if its curve is undefined (support 0 or all-positive),
    same guard as plot_roc_pr_curves."""
    curves = {}
    for i, name in enumerate(class_names):
        y_true, y_prob = labels_flat[:, i], probs_flat[:, i]
        support = int(y_true.sum())
        if support == 0 or support == len(y_true):
            curves[name] = None
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        curves[name] = (recall, precision, ap)
    return curves


def build_report_rows(dataset, indices, probs, class_names):
    """Maps each sample index to (idx, kind, class_idx) via the existing
    class_index_for_sample -- the same class-selection logic
    plot_tile_grid_overlay already uses, reused here so both figures agree
    on which class each sample is "about"."""
    rows = []
    for idx in indices:
        row = dataset.rows[idx]
        kind, class_idx = class_index_for_sample(row, class_names, probs[idx])
        rows.append((idx, kind, class_idx))
    return rows


def select_report_rows(dataset, probs, class_names, per_class=3, seed=0):
    """Picks which (sample, class) rows populate the per-sample report
    figure (plot_stage_b_eight_column_report): a diverse set covering every
    label kind (reusing select_diverse_sample_indices), each keyed to a
    class via build_report_rows."""
    indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=per_class, seed=seed)
    return build_report_rows(dataset, indices, probs, class_names)


def _clean_image_lookup(dataset, class_names):
    """source_id -> its clean-variant row (every class inactive) within the
    same split as `dataset` -- guaranteed present per build_stage_b_dataset's
    balanced construction (exactly one clean variant per source image, same
    split as its distorted variants). Returns {} gracefully if none exist
    (e.g. a hand-built test fixture with no clean row)."""
    return {
        row["source_id"]: row
        for row in dataset.rows
        if not any(int(row[c]) for c in class_names)
    }


def _load_image_for_row(dataset, row):
    """Loads+preprocesses one metadata row's image the same way
    StageBDataset.__getitem__ does (BGR->RGB, resize to dataset.img_size),
    returning a uint8 HWC array directly -- used for the clean-reference
    image, which may not be at a known dataset[] index."""
    image = cv.imread(str(dataset.data_dir / row["path"]))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (dataset.img_size, dataset.img_size))
    return image


def plot_stage_b_eight_column_report(
    dataset, report_rows, probs, labels, class_names,
    labels_flat, probs_flat, out_dir, tag="", rows_per_page=6,
    gate_probs=None, gated_tile_probs=None, gate_threshold=0.5,
):
    """Stage B results figure (supervisor item 1, extended per two
    follow-ups): one row per (idx, kind, class_idx) entry in `report_rows`.
    Originally 5 columns with a single "predicted probability" column for
    just the row's dominant/active class; extended to show all 3 classes'
    tile-wise probability side by side (+2 columns), then again to add a
    dedicated gated/final-decision column (+1) -- 8 total:
      1. Original (clean) image for that sample's source image.
      2. Distorted image -- the actual dataset[idx] model input.
      3. Ground truth tile grid for class_idx (the row's dominant/active
         class only -- the supervisor's asks were specifically about the
         probability columns, not GT), with a 0-1 colorbar.
      4-6. Predicted tile probability, RAW (not thresholded, never gated),
         one column per class in `class_names` order (not just class_idx)
         -- same 0-1 colorbar style. The dominant class's column is marked
         with a "(dominant)" title suffix so it's identifiable among the
         three.
      7. The dominant class's precision-recall curve (AUC-PR), computed
         once over the whole flattened test split and reused for every row
         of the same class -- it is not a per-sample quantity.
      8. The dominant class's GATED tile probability, i.e. what columns
         4-6's dominant-class panel looks like AFTER
         `src.eval.gate.apply_gate` -- all-zero if the gate called this
         image "not impaired". Lets a reader directly compare the raw
         prediction (columns 4-6) against the final, post-gate result.
         Falls back to the same raw grid (labeled "no gate applied") when
         `gated_tile_probs` isn't given.

    Columns 3-6 and 8 deliberately use plain imshow(cmap=..., vmin=0,
    vmax=1) + colorbar -- NOT plot_tile_grid_overlay's alpha-blended
    composite-over-image style, which can't be legended by a single 0-1
    colorbar. See that function's own docstring/caption for why it's left
    unchanged instead of migrated to this style.

    `gate_probs` (optional dict[idx -> P(impaired)] from ImpairedGateHead,
    see src.eval.gate.collect_gate_probs) annotates each row's "distorted"
    column title with the gate's own verdict. `probs` (used for columns
    4-6 and the PR curve) should always be the RAW (ungated) predictions --
    a gated row previously showed all-zero tiles there, which hid whether
    the gate was right to suppress it or just killed a real detection.
    `gated_tile_probs` (optional, the full (N,C,H,W) array after
    `apply_gate`) supplies column 8's post-gate view instead; `main()`
    still applies `apply_gate` for the *other* outputs this script writes
    (metrics table, ROC/PR curves file, overlay grid, severity plot) using
    the same gated array.

    Paginated at `rows_per_page` rows per file:
    out_dir/stage_b_full_report{tag}_page{N}.jpg (N starting at 1, always at
    least one page even for an empty report_rows). Returns the list of
    written paths."""
    n_classes = len(class_names)
    n_cols = 3 + n_classes + 2  # original, distorted, GT, one prob column per class, PR curve, gated
    curves = _compute_per_class_pr_curves(labels_flat, probs_flat, class_names)
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

            gt_grid = labels[idx, class_idx].astype(np.float32)
            im_gt = axes[r, 2].imshow(gt_grid, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            axes[r, 2].set_title(f"GT tiles ({class_name})", fontsize=9)
            axes[r, 2].set_xticks([])
            axes[r, 2].set_yticks([])
            fig.colorbar(im_gt, ax=axes[r, 2], fraction=0.046)

            for j, other_name in enumerate(class_names):
                col = 3 + j
                pred_grid = probs[idx, j].astype(np.float32)
                im_pred = axes[r, col].imshow(pred_grid, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                title = f"predicted probability ({other_name})"
                if j == class_idx:
                    title += "\n(dominant)"
                axes[r, col].set_title(title, fontsize=9)
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])
                fig.colorbar(im_pred, ax=axes[r, col], fraction=0.046)

            pr_col = 3 + n_classes
            curve = curves.get(class_name)
            if curve is None:
                axes[r, pr_col].text(
                    0.5, 0.5, "PR curve undefined\n(no positives in split)",
                    ha="center", va="center", fontsize=9,
                )
                axes[r, pr_col].axis("off")
            else:
                recall, precision, ap = curve
                axes[r, pr_col].plot(recall, precision)
                axes[r, pr_col].set_xlim(0, 1)
                axes[r, pr_col].set_ylim(0, 1.02)
                axes[r, pr_col].set_xlabel("recall", fontsize=8)
                axes[r, pr_col].set_ylabel("precision", fontsize=8)
                axes[r, pr_col].set_title(f"{class_name} PR (AP={ap:.3f})", fontsize=9)

            gated_col = 3 + n_classes + 1
            if gated_tile_probs is not None:
                gated_grid = gated_tile_probs[idx, class_idx].astype(np.float32)
                gated_title = f"gated probability ({class_name})"
            else:
                gated_grid = probs[idx, class_idx].astype(np.float32)
                gated_title = f"gated probability ({class_name})\n(no gate applied)"
            im_gated = axes[r, gated_col].imshow(gated_grid, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            axes[r, gated_col].set_title(gated_title, fontsize=9)
            axes[r, gated_col].set_xticks([])
            axes[r, gated_col].set_yticks([])
            fig.colorbar(im_gated, ax=axes[r, gated_col], fraction=0.046)

        for r in range(len(page_rows), n_rows):
            for c in range(n_cols):
                axes[r, c].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"stage_b_full_report{tag}_page{page_num}.jpg"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/stage_b/stage_b_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Use per-class best-F1 thresholds tuned on the val split (same as "
                        "evaluate_stage_b.py --tune-thresholds) instead of --threshold, so the figures "
                        "match the evaluation table.")
    parser.add_argument("--per-kind", type=int, default=3, help="Sample images per label kind for the grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--log-file", default=None, help="Saved training stdout log (e.g. stage_b_<jobid>.out) -- if given, also plots the train/val loss curve")
    parser.add_argument("--tag", default="", help="Suffix (e.g. '_focal') appended to every output filename, so multiple loss variants don't overwrite each other's images")
    parser.add_argument("--rows-per-page", type=int, default=6, help="Rows per page in the 8-column report figure")
    parser.add_argument(
        "--gate-checkpoint", default=None,
        help="Optional checkpoints/impaired_gate/impaired_gate_head.pt -- if given, ACTUALLY GATES every "
        "figure/metric below: Stage B's tile predictions are zeroed for any image the gate calls "
        "'not impaired' (Session 20, Round 2 -- see src/eval/gate.py). The report renders fine, ungated, "
        "without this flag.",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5,
        help="P(impaired) cutoff for --gate-checkpoint: below this, an image's Stage B predictions are zeroed.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    backbone, head, class_names = load_stage_b(args.checkpoint, args.weights, device)

    threshold = args.threshold
    if args.tune_thresholds:
        val_loader = DataLoader(StageBDataset(args.data, split="val"), batch_size=32, shuffle=False)
        val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
        threshold = tune_per_class_thresholds(*flatten_tiles(val_labels, val_probs), class_names)
        print(f"tuned thresholds: {threshold}")

    dataset = StageBDataset(args.data, split=args.split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    raw_probs, labels = collect_predictions(backbone, head, loader, device)
    probs = raw_probs

    gate_probs = None
    if args.gate_checkpoint:
        gate_head, gate_img_size = load_gate(args.gate_checkpoint, backbone.out_channels, device)
        gate_probs = collect_gate_probs(backbone, gate_head, loader, device, img_size=gate_img_size)
        # Real inference-time gate (Session 20, Round 2): the metrics table,
        # ROC/PR curves file, overlay grid, and severity plot below all see
        # the GATED probs -- `apply_gate` returns a new array, so `raw_probs`
        # still holds the ungated predictions afterward. The 8-column report
        # (Round 3 follow-ups) deliberately uses `raw_probs` for columns 4-6
        # and passes this gated array separately for column 8 -- see its own
        # docstring for why.
        probs = apply_gate(raw_probs, gate_probs, threshold=args.gate_threshold)

    labels_flat, probs_flat = flatten_tiles(labels, probs)
    # gating never touches `labels`, only `probs` -- so flattened labels are
    # identical either way; only the raw probs differ from the gated ones.
    _, raw_probs_flat = flatten_tiles(labels, raw_probs)
    metric_rows = compute_metrics(labels_flat, probs_flat, class_names, threshold=threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"stage_b_test_metrics{args.tag}.jpg"
    plot_metrics_bar_chart(metric_rows, metrics_path, title=f"Stage B per-tile metrics per class{args.tag}")
    print(f"wrote {metrics_path}")

    curves_path = out_dir / f"stage_b_roc_pr_curves{args.tag}.jpg"
    plot_roc_pr_curves(labels_flat, probs_flat, class_names, curves_path, title=f"Stage B{args.tag}")
    print(f"wrote {curves_path}")

    severity_path = out_dir / f"stage_b_probability_by_severity{args.tag}.jpg"
    plot_probability_by_severity(dataset.rows, max_prob_per_image(probs), class_names, severity_path,
                                  title=f"Stage B{args.tag} (max prob per image)")
    print(f"wrote {severity_path}")

    sample_indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=args.per_kind, seed=args.seed)
    grid_path = out_dir / f"stage_b_sample_predictions{args.tag}.jpg"
    plot_tile_grid_overlay(dataset, sample_indices, probs, labels, class_names, threshold, grid_path)
    print(f"wrote {grid_path}")

    combo_idx, combo_classes = find_combo_sample_index(dataset.rows, class_names)
    if combo_idx is not None:
        combo_path = out_dir / f"stage_b_combo_sample{args.tag}.jpg"
        plot_combo_sample(dataset, combo_idx, combo_classes, probs, labels, class_names, threshold, combo_path)
        print(f"wrote {combo_path}")

    # Round 3 follow-up: the report's columns 4-6 (and PR curve) always show
    # the RAW (ungated) per-tile predictions, even when --gate-checkpoint is
    # given -- so a reader can visually judge whether a gate "not impaired"
    # verdict actually suppressed a real detection or correctly caught a
    # clean image. Column 8 shows the gated/final result for direct
    # comparison. Every other output this script writes above still uses
    # the gated `probs`.
    report_rows = select_report_rows(dataset, raw_probs, class_names, per_class=args.per_kind, seed=args.seed)
    report_paths = plot_stage_b_eight_column_report(
        dataset, report_rows, raw_probs, labels, class_names,
        labels_flat, raw_probs_flat, out_dir, tag=args.tag,
        rows_per_page=args.rows_per_page, gate_probs=gate_probs,
        gated_tile_probs=probs if gate_probs is not None else None,
        gate_threshold=args.gate_threshold,
    )
    for p in report_paths:
        print(f"wrote {p}")

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
