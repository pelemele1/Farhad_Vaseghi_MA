"""
Generate the visual results for the Stage A final report
(docs/stage_a_final_report.md): a per-class metrics bar chart, a grid of
real predictions on test-split images (covering each label kind --
clean/dirt/water/scratch), and -- given a saved training log via
--log-file -- the train/val loss curve.

Usage:
    python scripts/visualize_stage_a_results.py \
        --checkpoint checkpoints/stage_a/stage_a_head.pt \
        --data data/processed/stage_a --split test --device cpu \
        --log-file stage_a_<jobid>.out --out-dir docs/images
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_stage_a import collect_predictions, compute_metrics
from src.data.stage_a_dataset import StageADataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead


def select_diverse_sample_indices(rows, class_names, per_kind=3, seed=0):
    """rows: a list of metadata dicts (e.g. dataset.rows), in the same order
    as the model's predictions. Groups by label kind (clean, or whichever
    single class is active -- the dataset never combines two), then samples
    up to `per_kind` indices from each kind that actually occurs, so the
    resulting grid represents every kind rather than being purely random."""
    by_kind = {}
    for idx, row in enumerate(rows):
        active = [c for c in class_names if int(row[c])]
        kind = active[0] if active else "clean"
        by_kind.setdefault(kind, []).append(idx)

    rng = np.random.default_rng(seed)
    selected = []
    for kind in sorted(by_kind):
        indices = by_kind[kind]
        chosen = rng.choice(indices, size=min(per_kind, len(indices)), replace=False)
        selected.extend(int(i) for i in chosen)
    return selected


def kind_from_scores(scores, class_names, threshold=None):
    """scores: 1D array, either 0/1 ground-truth labels or [0,1] predicted
    probabilities (in which case `threshold` picks which count as active).
    Returns a tuple of active class names, or ("clean",) if none."""
    if threshold is None:
        active = tuple(name for name, v in zip(class_names, scores) if v)
    else:
        active = tuple(name for name, v in zip(class_names, scores) if v >= threshold)
    return active if active else ("clean",)


_EPOCH_LINE_RE = re.compile(
    r"^epoch (?P<epoch>\d+)/\d+\s+train_loss=(?P<train_loss>[\d.]+)\s+val_loss=(?P<val_loss>[\d.]+)"
)


def parse_training_log(log_text):
    """Extracts (epoch, train_loss, val_loss) triples from train_stage_a.py's
    stdout (e.g. 'epoch 3/20  train_loss=0.3027  val_loss=0.2884  (36.9s)'),
    in the order the lines appear. Any line not matching that exact format
    (SLURM prologue/epilogue, warnings, nvidia-smi output, etc.) is ignored,
    not treated as an error -- a raw job log is full of unrelated lines."""
    records = []
    for line in log_text.splitlines():
        m = _EPOCH_LINE_RE.match(line.strip())
        if m:
            records.append({
                "epoch": int(m.group("epoch")),
                "train_loss": float(m.group("train_loss")),
                "val_loss": float(m.group("val_loss")),
            })
    return records


def plot_training_curve(records, out_path, title="Stage A training curve (real TinyGPU run)", ylabel="BCE-with-logits loss"):
    epochs = [r["epoch"] for r in records]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_loss"] for r in records], marker="o", label="train loss")
    ax.plot(epochs, [r["val_loss"] for r in records], marker="o", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_metrics_bar_chart(metric_rows, out_path, title="Stage A test-split metrics per class"):
    names = [r["class"] for r in metric_rows]
    metrics = ["precision", "recall", "f1", "ap"]
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [r[m] for r in metric_rows], width, label=m.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_prediction_grid(dataset, indices, probs, labels, class_names, threshold, out_path, cols=4):
    n = len(indices)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4.6 * rows_n))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        image_tensor, _ = dataset[idx]
        image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ax.imshow(image)
        ax.axis("off")

        true_kind = kind_from_scores(labels[idx], class_names)
        pred_kind = kind_from_scores(probs[idx], class_names, threshold=threshold)
        correct = set(true_kind) == set(pred_kind)
        prob_str = "  ".join(f"{name}={p:.2f}" for name, p in zip(class_names, probs[idx]))

        ax.set_title(
            f"true: {'+'.join(true_kind)}   pred: {'+'.join(pred_kind)}\n{prob_str}",
            fontsize=10, color=("seagreen" if correct else "crimson"),
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/stage_a/stage_a_head.pt")
    parser.add_argument("--data", default="data/processed/stage_a")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--per-kind", type=int, default=3, help="Sample images per label kind for the grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--log-file", default=None, help="Saved training stdout log (e.g. stage_a_<jobid>.out) -- if given, also plots the train/val loss curve")
    parser.add_argument("--tag", default="", help="Suffix (e.g. '_combo') appended to every output filename, "
                         "so multiple checkpoints/datasets don't overwrite each other's images "
                         "(same convention as visualize_stage_b_results.py)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageADistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    dataset = StageADataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs, labels = collect_predictions(backbone, head, loader, device)
    metric_rows = compute_metrics(labels, probs, class_names, threshold=args.threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"stage_a_test_metrics{args.tag}.jpg"
    plot_metrics_bar_chart(metric_rows, metrics_path)
    print(f"wrote {metrics_path}")

    sample_indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=args.per_kind, seed=args.seed)
    grid_path = out_dir / f"stage_a_sample_predictions{args.tag}.jpg"
    plot_prediction_grid(dataset, sample_indices, probs, labels, class_names, args.threshold, grid_path)
    print(f"wrote {grid_path}")

    if args.log_file:
        records = parse_training_log(Path(args.log_file).read_text())
        if not records:
            print(f"warning: no 'epoch N/M train_loss=... val_loss=...' lines found in {args.log_file}")
        else:
            curve_path = out_dir / f"stage_a_training_curve{args.tag}.jpg"
            plot_training_curve(records, curve_path)
            print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
