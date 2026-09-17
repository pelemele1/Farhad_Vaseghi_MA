"""
Session 18: one qualitative "clean vs. distorted" figure per class, in the
style of a reference figure the user shared from an unrelated project --
original image and its distorted counterpart side by side, each with a
horizontal bar chart underneath showing the trained Stage A model's
predicted probability for every class on that specific image.

Bars are colored by outcome against ground truth, at the given threshold:
  - green  = hit            (ground truth positive, predicted positive)
  - red    = miss           (ground truth positive, predicted negative)
  - orange = false alarm    (ground truth negative, predicted positive)
  - gray   = correct reject (ground truth negative, predicted negative)

For each class, the clean/distorted pair is the *same underlying source
image* (same source_id in metadata.csv, one variant clean, one variant
positive for exactly that class) -- so the only difference between the two
photos is the synthetic distortion itself, matching the reference figure's
layout.

Usage:
    python scripts/visualize_stage_a_class_examples.py \
        --checkpoint checkpoints/stage_a/stage_a_head.pt \
        --data data/processed/stage_a --split test --device cpu \
        --out-dir docs/images
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.stage_a_dataset import StageADataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead


def find_clean_and_distorted_pair(rows, class_names, target_class, seed=0):
    """Finds one source_id (in `rows`, already filtered to one split) that
    has both a clean variant (all class labels 0) and a variant positive
    for exactly `target_class` (and no other class -- this project never
    combines distortions in one variant). Returns (clean_idx, distorted_idx)
    as indices into `rows`, or None if no such pair exists in this split."""
    by_source = {}
    for idx, row in enumerate(rows):
        by_source.setdefault(row["source_id"], []).append(idx)

    rng = np.random.default_rng(seed)
    source_ids = list(by_source)
    rng.shuffle(source_ids)

    for source_id in source_ids:
        indices = by_source[source_id]
        clean_idx, distorted_idx = None, None
        for idx in indices:
            row = rows[idx]
            active = [c for c in class_names if int(row[c])]
            if not active and clean_idx is None:
                clean_idx = idx
            elif active == [target_class] and distorted_idx is None:
                distorted_idx = idx
        if clean_idx is not None and distorted_idx is not None:
            return clean_idx, distorted_idx
    return None


@torch.no_grad()
def predict_one(backbone, head, dataset, idx, device):
    image_tensor, label = dataset[idx]
    logits = head(backbone(image_tensor.unsqueeze(0).to(device)))
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return image, label.numpy(), probs


def _bar_colors(labels, probs, threshold):
    colors = []
    for y_true, p in zip(labels, probs):
        pred = p >= threshold
        if y_true and pred:
            colors.append("seagreen")      # hit
        elif y_true and not pred:
            colors.append("crimson")       # miss
        elif not y_true and pred:
            colors.append("darkorange")    # false alarm
        else:
            colors.append("lightgray")     # correct reject
    return colors


def plot_class_example(class_names, target_class,
                        clean_image, clean_label, clean_probs,
                        dist_image, dist_label, dist_probs,
                        threshold, out_path):
    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.35, wspace=0.25)

    ax_img_clean = fig.add_subplot(gs[0, 0])
    ax_img_dist = fig.add_subplot(gs[0, 1])
    ax_bar_clean = fig.add_subplot(gs[1, 0])
    ax_bar_dist = fig.add_subplot(gs[1, 1])

    ax_img_clean.imshow(clean_image)
    ax_img_clean.set_title("Original (clean)")
    ax_img_clean.axis("off")

    ax_img_dist.imshow(dist_image)
    ax_img_dist.set_title(f"Synthetic — {target_class}")
    ax_img_dist.axis("off")

    y = np.arange(len(class_names))
    for ax, label, probs, subtitle in (
        (ax_bar_clean, clean_label, clean_probs, "Prediction — original"),
        (ax_bar_dist, dist_label, dist_probs, "Prediction — synthetic"),
    ):
        colors = _bar_colors(label, probs, threshold)
        ax.barh(y, probs, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(class_names)
        ax.set_xlim(0, 1.0)
        ax.axvline(threshold, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("predicted probability")
        ax.set_title(subtitle, fontsize=10)
        for yi, p in zip(y, probs):
            ax.text(min(p + 0.02, 0.9), yi, f"{p:.2f}", va="center", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="seagreen", label="hit"),
        plt.Rectangle((0, 0), 1, 1, color="lightgray", label="correct reject"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange", label="false alarm"),
        plt.Rectangle((0, 0), 1, 1, color="crimson", label="miss"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Stage A qualitative example — {target_class}", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="docs/images")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageADistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    dataset = StageADataset(args.data, split=args.split, img_size=args.img_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_class in class_names:
        pair = find_clean_and_distorted_pair(dataset.rows, class_names, target_class, seed=args.seed)
        if pair is None:
            print(f"skipping {target_class}: no clean+distorted pair from the same source image found in split={args.split}")
            continue
        clean_idx, distorted_idx = pair

        clean_image, clean_label, clean_probs = predict_one(backbone, head, dataset, clean_idx, device)
        dist_image, dist_label, dist_probs = predict_one(backbone, head, dataset, distorted_idx, device)

        out_path = out_dir / f"stage_a_example_{target_class}.jpg"
        plot_class_example(
            class_names, target_class,
            clean_image, clean_label, clean_probs,
            dist_image, dist_label, dist_probs,
            args.threshold, out_path,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
