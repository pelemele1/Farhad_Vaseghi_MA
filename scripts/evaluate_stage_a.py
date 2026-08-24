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

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.stage_a_dataset import StageADataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead


@torch.no_grad()
def collect_predictions(backbone, head, loader, device):
    """Runs the whole loader through backbone -> head -> sigmoid. Returns
    (probs, labels), both numpy arrays of shape (n_samples, n_classes)."""
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = head(backbone(images))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(labels, probs, class_names, threshold=0.5):
    """labels, probs: (n_samples, n_classes) arrays. Returns one dict per
    class with precision/recall/f1 (at `threshold`) and threshold-
    independent average precision (AP), plus the positive-class support
    count. AP is left as NaN for a class with zero positives in this split
    -- it isn't a meaningful score without at least one positive example.

    Computes each class's precision/recall/f1 independently via
    average="binary" on that one column, rather than calling
    precision_recall_fscore_support(..., average=None) on the whole
    (n_samples, n_classes) array at once: sklearn infers multilabel vs.
    plain binary classification from the *array shape*, and a single-class
    evaluation (one column) gets misread as ordinary 2-class binary
    classification instead of 1-class multilabel, silently shifting what
    each output index means. Doesn't happen with >=2 classes (our actual
    3-class case), but per-column is correct regardless of class count."""
    preds = (probs >= threshold).astype(int)
    rows = []
    for i, name in enumerate(class_names):
        y_true, y_pred, y_prob = labels[:, i], preds[:, i], probs[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        support = int(y_true.sum())
        ap = average_precision_score(y_true, y_prob) if support > 0 else float("nan")
        rows.append({
            "class": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
            "support": support,
        })
    return rows


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
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageADistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    dataset = StageADataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs, labels = collect_predictions(backbone, head, loader, device)
    rows = compute_metrics(labels, probs, class_names, threshold=args.threshold)

    print(f"Evaluated {len(dataset)} images from split={args.split!r}, threshold={args.threshold}")
    print(f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'AP':>10}{'support':>10}")
    for row in rows:
        print(f"{row['class']:<10}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['ap']:>10.3f}{row['support']:>10}")


if __name__ == "__main__":
    main()
