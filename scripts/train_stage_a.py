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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.stage_a_dataset import StageADataset
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageADistortionHead
from src.models.losses import build_stage_a_loss, compute_pos_weight
from src.soiling.dataset_builder import EFFECT_NAMES


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


def fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, epochs,
        ckpt_path=None, extra_ckpt=None):
    """Trains for `epochs`, printing one `epoch N/M train_loss=... val_loss=...`
    line per epoch (the format visualize_*_results.py's parse_training_log
    reads). If `ckpt_path` is given, saves the head whenever val loss
    improves -- the best-val epoch, not whatever the last epoch happens to
    be. `extra_ckpt` is merged into the saved dict (e.g. architecture
    metadata). Returns the best val loss."""
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = run_epoch(backbone, head, train_loader, device, loss_fn, optimizer)
        val_loss = run_epoch(backbone, head, val_loader, device, loss_fn, optimizer=None)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names,
                            "epoch": epoch, "val_loss": val_loss, **(extra_ckpt or {})}, ckpt_path)
                marker = "  *saved"
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({time.time() - start:.1f}s){marker}", flush=True)
    if ckpt_path is not None:
        print(f"wrote {ckpt_path} (best val_loss={best_val:.4f})", flush=True)
    return best_val


def low_severity_sample_weights(rows, low_weight):
    """Per-row sampling weight: `low_weight` for a row with at least one
    low-severity distortion, 1.0 otherwise -- shows the head faint examples
    more often without changing any label."""
    return [low_weight if any(r.get(f"{n}_severity") == "low" for n in EFFECT_NAMES) else 1.0
            for r in rows]


def make_loaders(train_set, val_set, batch_size, num_workers, low_severity_weight=1.0):
    """Worker processes decode/resize JPEGs in parallel -- with 0 workers the
    GPU sits idle waiting on single-threaded image loading. With
    `low_severity_weight` != 1, the train loader samples (with replacement,
    same epoch length) by `low_severity_sample_weights`; val is unaffected."""
    kwargs = {"num_workers": num_workers, "pin_memory": num_workers > 0,
              "persistent_workers": num_workers > 0}
    if low_severity_weight != 1.0:
        weights = low_severity_sample_weights(train_set.rows, low_severity_weight)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, **kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)


def add_common_args(parser):
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=0, help="torch seed (head init + shuffling)")
    parser.add_argument("--low-severity-weight", type=float, default=1.0,
                        help="Sample training images containing a low-severity distortion this many "
                        "times more often (1.0 = uniform shuffling)")


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

    train_set = StageADataset(args.data, split="train", img_size=args.img_size, max_samples=max_samples)
    val_set = StageADataset(args.data, split="val", img_size=args.img_size, max_samples=max_samples)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.num_workers,
                                            low_severity_weight=args.low_severity_weight)

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageADistortionHead(in_channels=backbone.out_channels).to(device)

    pos_weight = compute_pos_weight(Path(args.data) / "metadata.csv", split="train").to(device)
    loss_fn = build_stage_a_loss(pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"train={len(train_set)} val={len(val_set)} pos_weight={pos_weight.tolist()}", flush=True)

    ckpt_path = None if args.smoke_test else Path(args.out) / "stage_a_head.pt"
    fit(backbone, head, train_loader, val_loader, device, loss_fn, optimizer, args.epochs, ckpt_path)


if __name__ == "__main__":
    main()
