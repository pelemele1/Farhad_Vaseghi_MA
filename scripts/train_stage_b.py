"""
Train the Stage B tile-grid distortion head (architecture.md §2, §3 Option 1:
frozen backbone, only the head is optimized).

This script is meant to run on the FAU HPC (NHR@FAU TinyGPU); the smoke test
below is the only training this project runs from the local/dev side --
real training is the user's own `sbatch` job on their HPC allocation.

Usage:
    python scripts/train_stage_b.py --data data/processed/stage_b --epochs 20 --device cuda
    python scripts/train_stage_b.py --data data/processed/stage_b --epochs 40 --loss focal --device cuda

Smoke test (tiny subset, 1 epoch, CPU, no checkpoint written -- verifies the
pipeline runs, not a real training run):
    python scripts/train_stage_b.py --smoke-test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train_stage_a import add_common_args, fit, make_loaders
from src.data.stage_b_dataset import StageBDataset
from src.models.backbone import STRIDE_TAPS, FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead, StageBMultiScaleHead
from src.models.losses import (
    build_stage_b_focal_loss,
    build_stage_b_loss,
    build_stage_b_ssd_loss,
    compute_tile_pos_weight,
)

STAGE_B_ARCHS = ("conv1x1", "multiscale")


def build_stage_b_model(arch, weights, class_names, device):
    """(backbone, head) for a Stage B architecture: "conv1x1" (v1, one 1x1
    conv on P5) or "multiscale" (v2, stride-4/8/16 + P5 features pooled to
    the P5 tile grid). Checkpoints without an arch field are v1."""
    if arch == "multiscale":
        backbone = FrozenYOLOBackbone(weights, return_layers=[STRIDE_TAPS[s] for s in (4, 8, 16, 32)])
        head = StageBMultiScaleHead(backbone.out_channels, class_names=class_names)
    elif arch == "conv1x1":
        backbone = FrozenYOLOBackbone(weights)
        head = StageBDistortionHead(in_channels=backbone.out_channels, class_names=class_names)
    else:
        raise ValueError(f"unknown Stage B arch {arch!r}, expected one of {STAGE_B_ARCHS}")
    return backbone.to(device), head.to(device)


def load_stage_b(checkpoint, weights, device):
    """(backbone, head, class_names) from a Stage B checkpoint, head in eval mode."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone, head = build_stage_b_model(ckpt.get("arch", "conv1x1"), weights, class_names, device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return backbone, head, class_names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/stage_b", help="Stage B dataset dir (metadata.csv + images/ + tile_labels.npy)")
    parser.add_argument("--weights", default="weights/yolo11m.pt", help="COCO-pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=512, help="Must be a multiple of 32 -- also sets the tile grid size (img-size // 32)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/stage_b", help="Where to write the trained head's weights")
    parser.add_argument(
        "--loss", default="bce", choices=["bce", "focal", "ssd"],
        help="'bce': BCEWithLogitsLoss(pos_weight=...) (default, matches Stage A's approach). "
             "'focal': focal loss (Lin et al. 2017) -- an alternative for extreme class imbalance "
             "(architecture.md §4); see docs/development_log.md Session 17 for why this was added "
             "(the 'bce' path's pos_weight for Stage B's scratch class comes out ~95x, which "
             "empirically caused over-prediction rather than localization). "
             "'ssd': localized (per-tile) sum of squared differences between predicted probability "
             "and tile label, no imbalance correction -- see LocalizedSSDLoss in src/models/losses.py "
             "(supervisor request, Session 19+, compared against the focal alpha=0.75 winner).",
    )
    parser.add_argument(
        "--focal-alpha", default="0.25", help="Only used with --loss focal. Either a single float "
        "(applied to every class equally) or 3 comma-separated floats matching class order "
        "dirt,water,scratch (e.g. '0.75,0.75,0.9') for a per-class alpha -- see "
        "FocalLossWithLogits in src/models/losses.py for why per-class alpha is supported.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Only used with --loss focal")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Tiny subset, 1 epoch, CPU, no checkpoint written -- pipeline correctness only",
    )
    parser.add_argument("--arch", default="conv1x1", choices=STAGE_B_ARCHS,
                        help="'conv1x1': one 1x1 conv on P5 (v1). 'multiscale': stride-4/8/16 + P5 "
                        "features pooled to the tile grid (v2).")
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

    train_set = StageBDataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = StageBDataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.num_workers,
                                            low_severity_weight=args.low_severity_weight)

    backbone, head = build_stage_b_model(args.arch, args.weights, train_set.class_names, device)
    print(f"arch={args.arch}", flush=True)

    if args.loss == "focal":
        alpha_parts = [p.strip() for p in args.focal_alpha.split(",")]
        if len(alpha_parts) == 1:
            alpha = float(alpha_parts[0])
        else:
            assert len(alpha_parts) == len(train_set.class_names), (
                f"--focal-alpha has {len(alpha_parts)} comma-separated values but there are "
                f"{len(train_set.class_names)} classes {train_set.class_names} -- must match 1:1"
            )
            alpha = torch.tensor([float(p) for p in alpha_parts])
        loss_fn = build_stage_b_focal_loss(alpha=alpha, gamma=args.focal_gamma)
        alpha_str = alpha.tolist() if isinstance(alpha, torch.Tensor) else alpha
        print(f"train={len(train_set)} val={len(val_set)} loss=focal alpha={alpha_str} gamma={args.focal_gamma}")
    elif args.loss == "ssd":
        loss_fn = build_stage_b_ssd_loss()
        print(f"train={len(train_set)} val={len(val_set)} loss=ssd (localized sum of squared differences)")
    else:
        pos_weight = compute_tile_pos_weight(
            Path(args.data) / "tile_labels.npy", Path(args.data) / "metadata.csv", split="train"
        ).to(device)
        loss_fn = build_stage_b_loss(pos_weight)
        print(f"train={len(train_set)} val={len(val_set)} loss=bce pos_weight={pos_weight.tolist()}")

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    ckpt_path = None if args.smoke_test else Path(args.out) / "stage_b_head.pt"
    fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, args.epochs, ckpt_path,
        extra_ckpt={"arch": args.arch})


if __name__ == "__main__":
    main()
