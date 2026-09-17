"""
Evaluate a trained Stage A distortion head (architecture.md plan Step 6):
per-class precision/recall/F1/AP on a held-out split of the dataset built
by src/soiling/dataset_builder.py.

Usage:
    python scripts/evaluate_stage_a.py --checkpoint checkpoints/stage_a/stage_a_head.pt \
        --data data/processed/stage_a --split test --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.stage_a_dataset import StageADataset
from src.eval.metrics import collect_predictions, compute_metrics  # noqa: F401 (re-exported for tests/callers)
from src.eval.thresholds import tune_per_class_thresholds
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage_a/stage_a_head.pt")
    parser.add_argument("--data", default="data/processed/stage_a")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=640)
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
    head = StageADistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = StageADataset(args.data, split="val", img_size=args.img_size)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
        threshold = tune_per_class_thresholds(val_labels, val_probs, class_names)
        print("Tuned per-class thresholds (best-F1 on val split):")
        for name in class_names:
            print(f"  {name:<10}{threshold[name]:.3f}")

    dataset = StageADataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs, labels = collect_predictions(backbone, head, loader, device)
    rows = compute_metrics(labels, probs, class_names, threshold=threshold)

    threshold_desc = "tuned per-class" if args.tune_thresholds else args.threshold
    print(f"Evaluated {len(dataset)} images from split={args.split!r}, threshold={threshold_desc}")
    print(f"{'class':<10}{'threshold':>10}{'precision':>10}{'recall':>10}{'f1':>10}{'AP':>10}{'ROC-AUC':>10}{'support':>10}")
    for row in rows:
        print(f"{row['class']:<10}{row['threshold']:>10.3f}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['ap']:>10.3f}{row['roc_auc']:>10.3f}{row['support']:>10}")


if __name__ == "__main__":
    main()
