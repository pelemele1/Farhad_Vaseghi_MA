"""
Train the impaired-gate head (Session 20): an image-level binary "is this
image impaired at all" classifier, distinct from Stage A's per-defect-type
head, trained with CrossEntropyLoss on 2 mutually-exclusive classes
(not_impaired / impaired). Mirrors scripts/train_stage_a.py's structure and
reuses its run_epoch (fully generic over loss/label shape, no changes
needed).

Usage:
    python scripts/train_impaired_gate.py --data data/processed/stage_a --epochs 20 --device cuda

Smoke test (tiny subset, 1 epoch, CPU, no checkpoint written -- verifies the
pipeline runs, not a real training run):
    python scripts/train_impaired_gate.py --smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scripts.train_stage_a import run_epoch
from src.data.stage_a_dataset import ImpairedGateDataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import ImpairedGateHead
from src.models.losses import build_impaired_gate_loss, compute_impaired_class_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/stage_a", help="Stage A dataset dir (metadata.csv + images/)")
    parser.add_argument("--weights", default="weights/yolo11m.pt", help="COCO-pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/impaired_gate", help="Where to write the trained head's weights")
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

    train_set = ImpairedGateDataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = ImpairedGateDataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = ImpairedGateHead(in_channels=backbone.out_channels).to(device)

    class_weight = compute_impaired_class_weight(Path(args.data) / "metadata.csv", split="train").to(device)
    loss_fn = build_impaired_gate_loss(class_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"train={len(train_set)} val={len(val_set)} class_weight={class_weight.tolist()}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(backbone, head, train_loader, device, loss_fn, optimizer)
        val_loss = run_epoch(backbone, head, val_loader, device, loss_fn, optimizer=None)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({time.time() - start:.1f}s)")

    if not args.smoke_test:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "impaired_gate_head.pt"
        torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)
        print(f"wrote {ckpt_path}")


if __name__ == "__main__":
    main()
