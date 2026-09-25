"""
Evaluate a trained impaired-gate head (Session 20): image-level "is this
image impaired at all" binary classification. Mirrors
scripts/evaluate_stage_a.py, but needs a softmax- (not sigmoid-) based
prediction collector since ImpairedGateHead's 2 logits are mutually
exclusive, not per-class-independent.

Usage:
    python scripts/evaluate_impaired_gate.py --checkpoint checkpoints/impaired_gate/impaired_gate_head.pt \
        --data data/processed/stage_a --split test --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.stage_a_dataset import ImpairedGateDataset
from src.eval.metrics import compute_metrics
from src.eval.thresholds import tune_per_class_thresholds
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import ImpairedGateHead


@torch.no_grad()
def collect_gate_predictions(backbone, head, loader, device):
    """Like src.eval.metrics.collect_predictions, but applies softmax (not
    sigmoid) since ImpairedGateHead's 2 logits are mutually exclusive, and
    returns only the "impaired" (class-1) probability -- shaped (n, 1) so it
    drops straight into compute_metrics(..., class_names=["impaired"])
    unchanged."""
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(head(backbone(images)), dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)[:, None]
    labels = np.concatenate(all_labels)[:, None].astype(np.float32)
    return probs, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/impaired_gate/impaired_gate_head.pt")
    parser.add_argument("--data", default="data/processed/stage_a")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Softmax threshold for precision/recall/F1")
    parser.add_argument(
        "--tune-thresholds", action="store_true",
        help="Tune a best-F1 threshold on the val split, then evaluate --split with it "
        "instead of --threshold (see src/eval/thresholds.py).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    class_names = ["impaired"]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = ImpairedGateHead(in_channels=backbone.out_channels).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = ImpairedGateDataset(args.data, split="val", img_size=args.img_size)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        val_probs, val_labels = collect_gate_predictions(backbone, head, val_loader, device)
        threshold = tune_per_class_thresholds(val_labels, val_probs, class_names)
        print("Tuned threshold (best-F1 on val split):")
        for name in class_names:
            print(f"  {name:<10}{threshold[name]:.3f}")

    dataset = ImpairedGateDataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs, labels = collect_gate_predictions(backbone, head, loader, device)
    rows = compute_metrics(labels, probs, class_names, threshold=threshold)

    threshold_desc = "tuned" if args.tune_thresholds else args.threshold
    print(f"Evaluated {len(dataset)} images from split={args.split!r}, threshold={threshold_desc}")
    print(f"{'class':<10}{'threshold':>10}{'precision':>10}{'recall':>10}{'f1':>10}{'AP':>10}{'ROC-AUC':>10}{'support':>10}")
    for row in rows:
        print(f"{row['class']:<10}{row['threshold']:>10.3f}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['ap']:>10.3f}{row['roc_auc']:>10.3f}{row['support']:>10}")


if __name__ == "__main__":
    main()
