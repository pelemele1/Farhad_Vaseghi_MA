"""
Evaluate a trained Stage B tile-grid distortion head (architecture.md §2):
per-class precision/recall/F1/AP over individual *tiles* (not whole images)
on a held-out split of the dataset built by
src/soiling/dataset_builder.py::build_stage_b_dataset.

Usage:
    python scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b/stage_b_head.pt \
        --data data/processed/stage_b --split test --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.stage_b_dataset import StageBDataset
from src.eval.metrics import collect_predictions, compute_metrics
from src.eval.thresholds import tune_per_class_thresholds
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead


def flatten_tiles(labels, probs):
    """(N, C, H, W) -> (N*H*W, C): compute_metrics expects one row per
    sample, one column per class -- here a "sample" is one tile, not one
    image, so every tile of every image becomes its own row."""
    n, c, h, w = labels.shape
    labels_flat = labels.transpose(0, 2, 3, 1).reshape(-1, c)
    probs_flat = probs.transpose(0, 2, 3, 1).reshape(-1, c)
    return labels_flat, probs_flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage_b/stage_b_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for precision/recall/F1")
    parser.add_argument(
        "--tune-thresholds", action="store_true",
        help="Tune a per-class best-F1 threshold on the val split, then evaluate --split with those "
        "instead of the single --threshold value for every class (see src/eval/thresholds.py).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageBDistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = StageBDataset(args.data, split="val")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
        val_labels_flat, val_probs_flat = flatten_tiles(val_labels, val_probs)
        threshold = tune_per_class_thresholds(val_labels_flat, val_probs_flat, class_names)
        print("Tuned per-class thresholds (best-F1 on val split):")
        for name in class_names:
            print(f"  {name:<10}{threshold[name]:.3f}")

    # img_size intentionally not exposed here (unlike evaluate_stage_a.py) --
    # StageBDataset ties it to the dataset's own build-time img_size, since a
    # mismatch would misalign the backbone's feature-map grid against the
    # tile labels (see src/data/stage_b_dataset.py).
    dataset = StageBDataset(args.data, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs, labels = collect_predictions(backbone, head, loader, device)
    labels_flat, probs_flat = flatten_tiles(labels, probs)
    rows = compute_metrics(labels_flat, probs_flat, class_names, threshold=threshold)

    n_tiles = labels_flat.shape[0]
    threshold_desc = "tuned per-class" if args.tune_thresholds else args.threshold
    print(f"Evaluated {len(dataset)} images ({n_tiles} tiles) from split={args.split!r}, threshold={threshold_desc}")
    print(f"{'class':<10}{'threshold':>10}{'precision':>10}{'recall':>10}{'f1':>10}{'AP':>10}{'ROC-AUC':>10}{'support':>10}")
    for row in rows:
        print(f"{row['class']:<10}{row['threshold']:>10.3f}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['ap']:>10.3f}{row['roc_auc']:>10.3f}{row['support']:>10}")


if __name__ == "__main__":
    main()
