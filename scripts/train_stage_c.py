"""
Train the Stage C pixel-level segmentation head (architecture.md §2, §3
Option 1: frozen backbone, only the head is optimized).

This script is meant to run on the FAU HPC (NHR@FAU TinyGPU); the smoke test
below is the only training this project runs from the local/dev side --
real training is the user's own `sbatch` job on their HPC allocation.

Usage:
    python scripts/train_stage_c.py --data data/processed/stage_b --epochs 20 --device cuda

Smoke test (tiny subset, 1 epoch, CPU, no checkpoint written -- verifies the
pipeline runs, not a real training run):
    python scripts/train_stage_c.py --smoke-test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train_stage_a import add_common_args, fit, make_loaders
from src.data.stage_c_dataset import StageCDataset
from src.models.backbone import STRIDE_TAPS, FrozenYOLOBackbone
from src.models.distortion_head import StageCDistortionHead, StageCUNetHead
from src.models.losses import build_stage_c_loss

STAGE_C_ARCHS = ("unet", "fcn")


def build_stage_c_model(arch, weights, class_names, device):
    """(backbone, head) for a Stage C architecture: "unet" (v2, skip
    connections from stride 4/8/16 + P5) or "fcn" (v1, P5 only). Checkpoints
    written before the arch field existed are v1 -- load them with "fcn"."""
    if arch == "unet":
        backbone = FrozenYOLOBackbone(weights, return_layers=[STRIDE_TAPS[s] for s in (4, 8, 16, 32)])
        head = StageCUNetHead(backbone.out_channels, class_names=class_names)
    elif arch == "fcn":
        backbone = FrozenYOLOBackbone(weights)
        head = StageCDistortionHead(in_channels=backbone.out_channels, class_names=class_names)
    else:
        raise ValueError(f"unknown Stage C arch {arch!r}, expected one of {STAGE_C_ARCHS}")
    return backbone.to(device), head.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/stage_b",
                         help="Dataset dir built with --save-pixel-masks (metadata.csv + images/ + masks/)")
    parser.add_argument("--weights", default="weights/yolo11m.pt", help="COCO-pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=512, help="Must be a multiple of 32 (P5 stride)")
    parser.add_argument("--bce-weight", type=float, default=0.5,
                         help="DiceBCELoss's BCE/Dice mix -- 1.0 is pure BCE, 0.0 is pure soft Dice")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/stage_c", help="Where to write the trained head's weights")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Tiny subset, 1 epoch, CPU, no checkpoint written -- pipeline correctness only",
    )
    parser.add_argument("--arch", default="unet", choices=STAGE_C_ARCHS,
                         help="'unet': skip connections from stride 4/8/16 + P5 (v2). 'fcn': P5-only "
                         "5x upsample stack (v1).")
    add_common_args(parser)
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs, args.batch_size, args.num_workers = 1, 2, 0
        max_samples = 8
        print(f"[smoke-test] 1 epoch, batch_size=2, device={args.device}, 8 train / 8 val samples, no checkpoint written")
    else:
        max_samples = None

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_set = StageCDataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = StageCDataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.num_workers,
                                            low_severity_weight=args.low_severity_weight)

    backbone, head = build_stage_c_model(args.arch, args.weights, train_set.class_names, device)

    loss_fn = build_stage_c_loss(bce_weight=args.bce_weight)
    print(f"train={len(train_set)} val={len(val_set)} arch={args.arch} loss=dice_bce "
          f"bce_weight={args.bce_weight}", flush=True)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    ckpt_path = None if args.smoke_test else Path(args.out) / "stage_c_head.pt"
    fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, args.epochs, ckpt_path,
        extra_ckpt={"arch": args.arch})


if __name__ == "__main__":
    main()
