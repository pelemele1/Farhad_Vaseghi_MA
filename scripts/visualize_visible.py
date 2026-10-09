"""
Figures for the gate-first pipeline on the visible-change dataset (Session 27),
from scripts/evaluate_visible.py's JSON plus the checkpoints:
  visible_metrics.jpg          per-class metrics of Stages A, B and C (gated)
  visible_severity.jpg         recall per class and severity, all stages
  visible_confusion.jpg        Stage B / C confusion matrices (gated)
  visible_training_curves.jpg  train/val loss of the three heads (from job logs)
  visible_label_examples.jpg   visible-change ground truth on test images (no model)
  visible_report_pageN.jpg     per-image sheets: original | distorted + gate |
                               GT tiles | Stage B | GT pixels | Stage C | answer

Usage:
    python scripts/visualize_visible.py --results results/visible_eval.json \
        --a ... --b ... --c ... --config checkpoints/visible_pipeline.json \
        --logs a.out,b.out,c.out --out-dir docs/images
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from scripts.visualize_stage_a_results import parse_training_log
from src.data.visible_dataset import VisibleDataset
from src.pipeline import CLASS_COLORS, EFFECT_NAMES, DistortionPipeline, colorize, format_answer

LEVELS = ("low", "medium", "high")
METRIC_COLORS = {"precision": "#4c72b0", "recall": "#dd8452", "f1": "#55a868", "iou": "#8172b3", "ap": "#c44e52"}


def _bars(ax, rows, metrics, title):
    x = np.arange(len(rows))
    width = 0.8 / len(metrics)
    for j, m in enumerate(metrics):
        vals = [r[m] for r in rows]
        bars = ax.bar(x + (j - (len(metrics) - 1) / 2) * width, vals, width, label=m.upper() if m in ("ap", "iou")
                      else m, color=METRIC_COLORS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    rotation=90)
    ax.set_xticks(x, [r["class"] for r in rows])
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(metrics), fontsize=8, frameon=False)


def plot_metrics(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _bars(axes[0], results["stage_a"]["per_class"], ["precision", "recall", "f1", "ap"], "Stage A (image level)")
    _bars(axes[1], results["stage_b"]["gated"]["per_class"], ["precision", "recall", "f1", "iou", "ap"],
          "Stage B (32-px tiles, gated)")
    _bars(axes[2], results["stage_c"]["gated"]["per_class"], ["precision", "recall", "f1", "iou", "ap"],
          "Stage C (pixels, gated)")
    fig.suptitle("Test split: per-class metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def plot_severity(results, out_path):
    panels = [("Stage A (image)", results["stage_a"]["by_severity"]),
              ("Stage B (tiles, gated)", results["stage_b"]["gated"]["by_severity"]),
              ("Stage C (pixels, gated)", results["stage_c"]["gated"]["by_severity"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    shades = {"low": 0.45, "medium": 0.72, "high": 1.0}
    for ax, (title, by_sev) in zip(axes, panels):
        x = np.arange(len(EFFECT_NAMES))
        for j, lv in enumerate(LEVELS):
            vals = [by_sev.get(n, {}).get(lv, {}).get("recall", np.nan) for n in EFFECT_NAMES]
            colors = [np.array(CLASS_COLORS[n]) / 255 * shades[lv] + (1 - shades[lv]) for n in EFFECT_NAMES]
            bars = ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=colors, edgecolor="0.3", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, EFFECT_NAMES)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("recall")
    fig.legend(handles=[Patch(facecolor=str(1 - shades[lv] * 0.6), edgecolor="0.3", label=f"{lv} severity")
                        for lv in LEVELS], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Test split: recall by severity (bars left to right: low, medium, high)", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(results, out_path):
    names = ["clean"] + list(EFFECT_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, key) in zip(axes, [("Stage B (tiles, gated)", "stage_b"), ("Stage C (pixels, gated)", "stage_c")]):
        conf = np.array(results[key]["gated"]["confusion"], dtype=np.float64)
        norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("prediction")
        ax.set_ylabel("ground truth")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Test split: confusion matrices, each row normalized to 1", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_paths, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    titles = ["Stage A (BCE)", "Stage B (softmax focal)", "Stage C (cross-entropy + Dice)"]
    for ax, path, title in zip(axes, log_paths, titles):
        records = parse_training_log(Path(path).read_text(errors="ignore"))
        epochs = [r["epoch"] for r in records]
        ax.plot(epochs, [r["train_loss"] for r in records], marker="o", ms=3, label="train loss")
        ax.plot(epochs, [r["val_loss"] for r in records], marker="o", ms=3, label="val loss")
        best = min(records, key=lambda r: r["val_loss"])
        ax.axvline(best["epoch"], color="0.6", ls=":", label=f"best val (epoch {best['epoch']})")
        ax.set_xticks([e for e in epochs if e == 1 or e % 5 == 0])
        ax.set_xlabel("epoch")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("Training curves (visible-change labels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def select_report_samples(rows, seed=0):
    """Test images covering every kind: 2 clean, each distortion at each
    severity alone, and every combination."""
    rng = np.random.default_rng(seed)
    wanted = ["clean", "clean"] + [f"{n}:{lv}" for n in EFFECT_NAMES for lv in LEVELS]
    picks = []
    for kind in wanted:
        cands = [i for i, r in enumerate(rows) if r["kind"] == kind and i not in picks]
        if cands:
            picks.append(int(rng.choice(cands)))
    for combo in (("dirt", "water"), ("dirt", "scratch"), ("water", "scratch"), ("dirt", "water", "scratch")):
        cands = [i for i, r in enumerate(rows)
                 if tuple(t.split(":")[0] for t in r["kind"].split("+")) == combo and i not in picks]
        picks += [int(i) for i in rng.choice(cands, size=min(2 if len(combo) == 3 else 1, len(cands)),
                                             replace=False)]
    return picks


def plot_label_examples(ds, out_path, seed=1):
    """Distorted test image next to its visible-change ground truth, for the
    single distortions at every severity and the combinations."""
    rng = np.random.default_rng(seed)
    kinds = ["dirt:low", "water:low", "scratch:low", "dirt:high", "water:high", "scratch:high"]
    picks = [int(rng.choice([i for i, r in enumerate(ds.rows) if r["kind"] == k])) for k in kinds]
    for combo in (("dirt", "water"), ("dirt", "scratch"), ("water", "scratch"), ("dirt", "water", "scratch"),
                  ("dirt", "water"), ("dirt", "water", "scratch")):
        cands = [i for i, r in enumerate(ds.rows) if tuple(t.split(":")[0] for t in r["kind"].split("+")) == combo
                 and i not in picks]
        picks.append(int(rng.choice(cands)))
    fig, axes = plt.subplots(4, 6, figsize=(18, 12.6))
    for n, idx in enumerate(picks):
        r, c = divmod(n, 3)
        image_t, gt = ds[idx]
        img = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        row = ds.rows[idx]
        shares = ", ".join(f"{name} {100 * int(row[name + '_pixels']) / int(row['total_pixels']):.0f}%"
                           for name in EFFECT_NAMES if int(row[name]))
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title(row["kind"].replace("+", " + "), fontsize=9)
        axes[r, 2 * c + 1].imshow(colorize(gt.numpy(), img))
        axes[r, 2 * c + 1].set_title(f"label: {shares or 'clean'}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.legend(handles=[Patch(color=np.array(CLASS_COLORS[n]) / 255, label=n) for n in EFFECT_NAMES],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle("Visible-change ground truth (test images): one class per pixel, only what is visible", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def plot_report(pipeline, ds, indices, out_dir, rows_per_page=6, tag="visible"):
    tiles = VisibleDataset(ds.data_dir, "test", "tile").tile_labels
    clean_of = {r["source_id"]: i for i, r in enumerate(ds.rows) if r["kind"] == "clean"}
    col_titles = ["original", "distorted + gate (Stage A)", "ground truth (tiles)", "Stage B (tiles)",
                  "ground truth (pixels)", "Stage C (pixels)"]
    paths = []
    pages = [indices[i:i + rows_per_page] for i in range(0, len(indices), rows_per_page)]
    for page_num, page in enumerate(pages, start=1):
        fig, axes = plt.subplots(len(page), 7, figsize=(7 * 3.0, len(page) * 3.15), squeeze=False,
                                 gridspec_kw={"width_ratios": [1] * 6 + [0.75]})
        for r, idx in enumerate(page):
            row = ds.rows[idx]
            image_t, gt = ds[idx]
            pred = pipeline.predict(image_t[None])[0]
            img = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            clean = ((ds[clean_of[row["source_id"]]][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                     if row["source_id"] in clean_of else img)
            up = lambda m: cv.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
            panels = [clean, img, colorize(up(tiles[idx]), img), colorize(up(pred["tile_map"]), img),
                      colorize(gt.numpy(), img), colorize(pred["pixel_map"], img)]
            a_txt = "  ".join(f"{n[0].upper()} {p:.2f}" for n, p in zip(EFFECT_NAMES, pred["a_probs"]))
            titles = [f"original\ntruth: {row['kind'].replace('+', ' + ')}",
                      f"gate: {'IMPAIRED' if pred['gate_passed'] else 'clean'}\n{a_txt}"] + col_titles[2:]
            for c, (panel, title) in enumerate(zip(panels, titles)):
                axes[r, c].imshow(panel)
                axes[r, c].set_title(title, fontsize=8)
                axes[r, c].axis("off")
            truth = "clean" if row["dominant"] == "clean" else f"dominant: {row['dominant']}"
            axes[r, 6].axis("off")
            axes[r, 6].text(0.02, 0.5, "answer:\n" + format_answer(pred["answer"]) + f"\n\ntruth:\n{truth}",
                            fontsize=9, family="monospace", va="center",
                            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
        fig.legend(handles=[Patch(color=np.array(CLASS_COLORS[n]) / 255, label=n) for n in EFFECT_NAMES],
                   loc="lower center", ncol=3, fontsize=10, frameon=False)
        fig.suptitle(f"Gate-first pipeline on test images (page {page_num}/{len(pages)})", fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        path = Path(out_dir) / f"{tag}_report_page{page_num}.jpg"
        fig.savefig(path, dpi=70)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/visible_eval.json")
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--c")
    parser.add_argument("--config", default="checkpoints/visible_pipeline.json")
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--logs", default=None, help="Comma-separated training logs of stages a,b,c")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--tag", default="visible")
    parser.add_argument("--labels-only", action="store_true", help="Only the ground-truth example figure")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = VisibleDataset(args.data, "test", "pixel")
    plot_label_examples(ds, out / f"{args.tag}_label_examples.jpg")
    print(f"wrote {out / f'{args.tag}_label_examples.jpg'}")
    if args.labels_only:
        return
    results = json.loads(Path(args.results).read_text())
    written = []
    for name, fn in (("metrics", plot_metrics), ("severity", plot_severity), ("confusion", plot_confusion)):
        path = out / f"{args.tag}_{name}.jpg"
        fn(results, path)
        written.append(path)
    if args.logs:
        path = out / f"{args.tag}_training_curves.jpg"
        plot_training_curves(args.logs.split(","), path)
        written.append(path)

    pipeline = DistortionPipeline(args.a, args.b, args.c, args.weights, args.device, config=args.config)
    with torch.no_grad():
        written += plot_report(pipeline, ds, select_report_samples(ds.rows), out, tag=args.tag)
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
