"""
Trains one head on the visible-change dataset (Session 27):
  --stage a  image level: which distortions are visible (multi-label, BCE);
             also the pipeline's gate ("impaired" = any class present)
  --stage b  32-px tiles: one class per tile, clean/dirt/water/scratch (softmax focal loss)
  --stage c  pixels: one class per pixel (class-weighted cross-entropy + Dice)
All three read the same frozen YOLOv11-m backbone at 512 px, so the pipeline
runs the backbone once per image (src/pipeline.py).

Usage:
    python scripts/train_visible.py --stage b --data data/processed/visible \
        --epochs 40 --device cuda --out checkpoints/visible_b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train_stage_a import add_common_args, build_stage_a_model, fit, make_loaders
from scripts.train_stage_b import build_stage_b_model
from scripts.train_stage_c import build_stage_c_model
from src.data.visible_dataset import VisibleDataset
from src.models.backbone import parse_taps
from src.models.losses import (
    CEDiceLoss,
    MultiClassFocalLoss,
    build_stage_a_loss,
    compute_pos_weight,
    inverse_sqrt_frequency_weights,
)

STAGES = {
    # target, default taps, default epochs, hflip by default
    "a": ("image", "8,16,32", 20, False),
    "b": ("tile", "4,8,16,32", 40, True),
    "c": ("pixel", "2,4,8,16,32", 25, False),
}
CKPT_NAMES = {"a": "stage_a_head.pt", "b": "stage_b_head.pt", "c": "stage_c_head.pt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Default: 20 (a), 40 (b), 25 (c)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--taps", default=None, type=parse_taps, help="Default: 8,16,32 (a), 4-32 (b), 2-32 (c)")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Stage B head width (Session 25: 32)")
    parser.add_argument("--hflip", default=None, choices=("on", "off"), help="Default: on for b, off for a/c")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-test", action="store_true", help="8 images, 1 epoch, no checkpoint")
    add_common_args(parser)
    args = parser.parse_args()

    target, default_taps, default_epochs, default_flip = STAGES[args.stage]
    taps = args.taps or parse_taps(default_taps)
    epochs = args.epochs or default_epochs
    hflip = default_flip if args.hflip is None else args.hflip == "on"
    max_samples = None
    if args.smoke_test:
        epochs, args.batch_size, args.num_workers, max_samples = 1, 2, 0, 8

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_set = VisibleDataset(args.data, "train", target, args.img_size, max_samples, hflip=hflip)
    val_set = VisibleDataset(args.data, "val", target, args.img_size, max_samples)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.num_workers,
                                            low_severity_weight=args.low_severity_weight)
    meta = train_set.meta

    if args.stage == "a":
        backbone, head = build_stage_a_model(args.weights, train_set.effect_names, device, taps=taps)
        pos_weight = compute_pos_weight(Path(args.data) / "metadata.csv", split="train").to(device)
        loss_fn = build_stage_a_loss(pos_weight)
        loss_desc = f"bce pos_weight={[round(w, 3) for w in pos_weight.tolist()]}"
        extra = {}
    elif args.stage == "b":
        backbone, head = build_stage_b_model("multiscale", args.weights, train_set.class_names, device, taps=taps,
                                             hidden_dim=args.hidden_dim, fuse_kernel=3)
        weights = inverse_sqrt_frequency_weights(meta["tile_counts"]["train"])
        loss_fn = MultiClassFocalLoss(weights, gamma=args.focal_gamma).to(device)
        loss_desc = f"softmax focal gamma={args.focal_gamma} class_weights={[round(w, 3) for w in weights.tolist()]}"
        extra = {"arch": "multiscale", "hidden_dim": args.hidden_dim, "fuse_kernel": 3}
    else:
        backbone, head = build_stage_c_model("unet", args.weights, train_set.class_names, device, taps=taps)
        counts = [meta["pixel_counts"]["train"][n] for n in train_set.class_names]
        weights = inverse_sqrt_frequency_weights(counts)
        loss_fn = CEDiceLoss(weights, dice_weight=args.dice_weight).to(device)
        loss_desc = f"ce+dice dice_weight={args.dice_weight} class_weights={[round(w, 3) for w in weights.tolist()]}"
        extra = {"arch": "unet"}

    print(f"stage {args.stage}: target={target} taps={taps} hflip={hflip} epochs={epochs} "
          f"train={len(train_set)} val={len(val_set)}", flush=True)
    print(f"head parameters: {sum(p.numel() for p in head.parameters())}; loss: {loss_desc}", flush=True)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    ckpt_path = None if args.smoke_test else Path(args.out) / CKPT_NAMES[args.stage]
    fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, epochs, ckpt_path,
        extra_ckpt={"stage": args.stage, "taps": list(taps), "img_size": args.img_size, "hflip": hflip,
                    "label_rule": meta["label_rule"], **extra})
    if ckpt_path is not None:
        with open(Path(args.out) / "train_args.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}, f, indent=2)


if __name__ == "__main__":
    main()
