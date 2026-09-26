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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train_stage_a import add_common_args, fit, make_loaders
from src.data.stage_a_dataset import ImpairedGateDataset
from src.models.backbone import STRIDE_TAPS, FrozenYOLOBackbone
from src.models.distortion_head import ImpairedGateHead
from src.models.losses import build_impaired_gate_loss, compute_impaired_class_weight

GATE_ARCHS = ("p5", "multiscale")


def build_gate_model(arch, weights, device):
    if arch == "multiscale":
        backbone = FrozenYOLOBackbone(weights, return_layers=[STRIDE_TAPS[s] for s in (4, 8, 16, 32)])
    elif arch == "p5":
        backbone = FrozenYOLOBackbone(weights)
    else:
        raise ValueError(f"unknown gate arch {arch!r}, expected one of {GATE_ARCHS}")
    return backbone.to(device), ImpairedGateHead(in_channels=backbone.out_channels).to(device)


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
    parser.add_argument("--arch", default="p5", choices=GATE_ARCHS,
                        help="'p5': GAP over P5 only (v1). 'multiscale': GAP over the stride-4/8/16 layers "
                        "and P5, concatenated (v2).")
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

    train_set = ImpairedGateDataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = ImpairedGateDataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.num_workers,
                                            low_severity_weight=args.low_severity_weight)

    backbone, head = build_gate_model(args.arch, args.weights, device)
    print(f"arch={args.arch} img_size={args.img_size}", flush=True)

    class_weight = compute_impaired_class_weight(Path(args.data) / "metadata.csv", split="train").to(device)
    loss_fn = build_impaired_gate_loss(class_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"train={len(train_set)} val={len(val_set)} class_weight={class_weight.tolist()}", flush=True)

    ckpt_path = None if args.smoke_test else Path(args.out) / "impaired_gate_head.pt"
    fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, args.epochs, ckpt_path,
        extra_ckpt={"img_size": args.img_size, "arch": args.arch})


if __name__ == "__main__":
    main()
