"""
Train the Stage A distortion head (architecture.md §3 Option 1: frozen
backbone, only the head is optimized).

This script is meant to run on the FAU HPC (NHR@FAU TinyGPU); the smoke test
below is the only training this project runs from the local/dev side --
real training is the user's own `sbatch` job on their HPC allocation.

Usage:
    python scripts/train_stage_a.py --data data/processed/stage_a --epochs 20 --device cuda

Smoke test (tiny subset, 1 epoch, CPU, no checkpoint written -- verifies the
pipeline runs, not a real training run):
    python scripts/train_stage_a.py --smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.stage_a_dataset import StageADataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead
from src.models.losses import build_stage_a_loss, compute_pos_weight


def run_epoch(backbone, head, loader, device, loss_fn, optimizer=None):
    """One pass over `loader`. Trains the head if `optimizer` is given,
    otherwise just evaluates. Returns the mean per-batch loss."""
    head.train(optimizer is not None)
    total_loss, n_batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = backbone(images)
        logits = head(features)
        loss = loss_fn(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/stage_a", help="Stage A dataset dir (metadata.csv + images/)")
    parser.add_argument("--weights", default="weights/yolo11m.pt", help="COCO-pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/stage_a", help="Where to write the trained head's weights")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Tiny subset, 1 epoch, CPU, no checkpoint written -- pipeline correctness only",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs, args.batch_size = 1, 2
        max_samples = 8
        print(f"[smoke-test] 1 epoch, batch_size=2, device={args.device}, 8 train / 8 val samples, no checkpoint written")
    else:
        max_samples = None

    device = torch.device(args.device)

    train_set = StageADataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = StageADataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageADistortionHead(in_channels=backbone.out_channels).to(device)

    pos_weight = compute_pos_weight(Path(args.data) / "metadata.csv", split="train").to(device)
    loss_fn = build_stage_a_loss(pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"train={len(train_set)} val={len(val_set)} pos_weight={pos_weight.tolist()}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(backbone, head, train_loader, device, loss_fn, optimizer)
        val_loss = run_epoch(backbone, head, val_loader, device, loss_fn, optimizer=None)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({time.time() - start:.1f}s)")

    if not args.smoke_test:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "stage_a_head.pt"
        torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)
        print(f"wrote {ckpt_path}")


if __name__ == "__main__":
    main()
