import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses import (
    DiceBCELoss,
    FocalLossWithLogits,
    LocalizedSSDLoss,
    build_impaired_gate_loss,
    build_stage_a_loss,
    build_stage_b_focal_loss,
    build_stage_b_loss,
    build_stage_b_ssd_loss,
    build_stage_c_loss,
    compute_impaired_class_weight,
    compute_pos_weight,
    compute_tile_pos_weight,
)
from src.soiling.dataset_builder import EFFECT_NAMES


def _write_metadata(path, rows):
    fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_pos_weight_matches_hand_calculation(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s0", "variant_id": 1, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
        {"path": "c", "source_id": "s0", "variant_id": 2, "split": "train", "dirt": 0, "water": 1, "scratch": 0},
        {"path": "d", "source_id": "s0", "variant_id": 3, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    pos_weight = compute_pos_weight(csv_path)
    # dirt: 1 positive / 3 negative -> 3.0; water: same -> 3.0;
    # scratch: 0 positive / 4 negative -> 4/max(0,1) = 4.0
    assert torch.allclose(pos_weight, torch.tensor([3.0, 3.0, 4.0]))


def test_compute_pos_weight_respects_split_filter(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    pos_weight = compute_pos_weight(csv_path, split="train")
    # only the "train" row counted: dirt 1 positive/0 negative -> 0.0;
    # water/scratch 0 positive/1 negative -> 1.0
    assert torch.allclose(pos_weight, torch.tensor([0.0, 1.0, 1.0]))


def test_build_stage_a_loss_is_finite_and_uses_pos_weight():
    pos_weight = torch.tensor([2.0, 1.0, 3.0])
    loss_fn = build_stage_a_loss(pos_weight)
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)


# --- Impaired gate head loss (Session 20, image-level binary CE) --------


def test_compute_impaired_class_weight_matches_hand_calculation(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s0", "variant_id": 1, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
        {"path": "c", "source_id": "s0", "variant_id": 2, "split": "train", "dirt": 0, "water": 1, "scratch": 0},
        {"path": "d", "source_id": "s0", "variant_id": 3, "split": "train", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    weight = compute_impaired_class_weight(csv_path)
    # 4 rows total, 2 impaired (a, c), 2 not_impaired (b, d)
    # weight[0] = 4/2 = 2.0, weight[1] = 4/2 = 2.0
    assert torch.allclose(weight, torch.tensor([2.0, 2.0]))


def test_compute_impaired_class_weight_respects_split_filter(tmp_path):
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
        {"path": "c", "source_id": "s1", "variant_id": 1, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata(csv_path, rows)

    weight = compute_impaired_class_weight(csv_path, split="val")
    # only "val" rows counted: 2 rows, both not_impaired
    # weight[0] = 2/2 = 1.0, weight[1] = 2/max(0,1) = 2.0
    assert torch.allclose(weight, torch.tensor([1.0, 2.0]))


def test_build_impaired_gate_loss_is_finite():
    class_weight = torch.tensor([1.0, 3.0])
    loss_fn = build_impaired_gate_loss(class_weight)
    logits = torch.randn(5, 2)
    labels = torch.randint(0, 2, (5,), dtype=torch.long)
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)


# --- Stage B (tile-grid) -------------------------------------------------


def _write_metadata_rows(path, rows):
    fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_tile_pos_weight_matches_hand_calculation(tmp_path):
    # 2 rows, each a 2x2 grid (4 tiles). dirt: 1 positive tile total out of
    # 8 -> pos_weight = (8-1)/1 = 7.0. water: 0 positive -> (8-0)/max(0,1) = 8.0.
    # scratch: 4 positive (all of row 2's tiles) -> (8-4)/4 = 1.0.
    tile_labels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    tile_labels[0, 0, 0, 0] = 1  # one dirt tile in row 0
    tile_labels[1, 2, :, :] = 1  # all 4 scratch tiles in row 1
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "train", "dirt": 0, "water": 0, "scratch": 1},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_rows(csv_path, rows)
    npy_path = tmp_path / "tile_labels.npy"
    np.save(npy_path, tile_labels)

    pos_weight = compute_tile_pos_weight(npy_path, csv_path)
    assert torch.allclose(pos_weight, torch.tensor([7.0, 8.0, 1.0]))


def test_compute_tile_pos_weight_respects_split_filter(tmp_path):
    tile_labels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    tile_labels[0, 0, 0, 0] = 1
    rows = [
        {"path": "a", "source_id": "s0", "variant_id": 0, "split": "train", "dirt": 1, "water": 0, "scratch": 0},
        {"path": "b", "source_id": "s1", "variant_id": 0, "split": "val", "dirt": 0, "water": 0, "scratch": 0},
    ]
    csv_path = tmp_path / "metadata.csv"
    _write_metadata_rows(csv_path, rows)
    npy_path = tmp_path / "tile_labels.npy"
    np.save(npy_path, tile_labels)

    pos_weight = compute_tile_pos_weight(npy_path, csv_path, split="val")
    # only row "b" (all-zero, 4 tiles/class) counted -> (4-0)/max(0,1) = 4.0 for every class
    assert torch.allclose(pos_weight, torch.tensor([4.0, 4.0, 4.0]))


def test_build_stage_b_loss_is_finite_and_none_pos_weight_is_safe():
    loss_fn = build_stage_b_loss(None)
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 2, (2, 3, 4, 4)).float()
    assert torch.isfinite(loss_fn(logits, labels))


def test_build_stage_b_loss_reshapes_pos_weight_onto_the_channel_axis():
    # A (C,) pos_weight broadcasts against the *last* dim by default -- if W
    # also happens to equal C (here both 3, e.g. a 4x3 grid), an unreshaped
    # (C,) pos_weight would silently apply per-column instead of
    # per-channel, without even raising a shape error. Confirm
    # build_stage_b_loss's result matches the correctly-reshaped (C, 1, 1)
    # computation and differs from that naive (unreshaped) one, so this test
    # would fail if the reshape were removed.
    torch.manual_seed(0)
    pos_weight = torch.tensor([5.0, 1.0, 1.0])
    logits = torch.randn(2, 3, 4, 3)  # B, C=3, H=4, W=3 -- W matches C on purpose
    labels = torch.randint(0, 2, (2, 3, 4, 3)).float()

    correct = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(-1, 1, 1))(logits, labels)
    naive = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)  # wrong axis: applies along W, not C

    got = build_stage_b_loss(pos_weight)(logits, labels)
    assert torch.allclose(got, correct)
    assert not torch.allclose(got, naive)


# --- Focal loss (Stage B scratch-class fix, Session 17) -----------------


def test_focal_loss_is_finite_on_random_inputs():
    loss_fn = build_stage_b_focal_loss()
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 2, (2, 3, 4, 4)).float()
    assert torch.isfinite(loss_fn(logits, labels))


def test_focal_loss_downweights_confident_correct_predictions_more_than_uncertain_ones():
    # Two single-element cases, both correctly predicted positive (target=1):
    # one confidently (large positive logit -> p close to 1, "easy"), one
    # barely (logit near 0 -> p ~ 0.5, "hard"). gamma>0 should shrink the
    # easy case's loss much more (relative to gamma=0, i.e. plain
    # alpha-weighted BCE) than it shrinks the hard case's -- that's the
    # whole point of the (1 - p_t)^gamma focusing term.
    target = torch.tensor([[1.0]])
    easy_logit = torch.tensor([[6.0]])   # sigmoid(6) ~ 0.9975, confidently correct
    hard_logit = torch.tensor([[0.1]])   # sigmoid(0.1) ~ 0.525, barely correct

    no_focus = FocalLossWithLogits(alpha=0.25, gamma=0.0)
    with_focus = FocalLossWithLogits(alpha=0.25, gamma=2.0)

    easy_ratio = with_focus(easy_logit, target) / no_focus(easy_logit, target)
    hard_ratio = with_focus(hard_logit, target) / no_focus(hard_logit, target)
    assert easy_ratio < hard_ratio


def test_focal_loss_gamma_zero_matches_alpha_weighted_bce_by_hand():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    alpha = 0.3

    loss_fn = FocalLossWithLogits(alpha=alpha, gamma=0.0)
    got = loss_fn(logits, targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    expected = (alpha_t * bce).mean()
    assert torch.allclose(got, expected)


def test_focal_loss_per_class_alpha_broadcasts_onto_channel_axis():
    # Same W==C pitfall as the pos_weight broadcast test above: a (C,) alpha
    # tensor must land on the channel dim, not silently broadcast against a
    # same-sized last (W) dim.
    torch.manual_seed(0)
    alpha = torch.tensor([0.9, 0.1, 0.1])  # heavily favor class 0's positives
    logits = torch.randn(2, 3, 4, 3)  # B, C=3, H=4, W=3 -- W matches C on purpose
    labels = torch.randint(0, 2, (2, 3, 4, 3)).float()

    loss_fn = FocalLossWithLogits(alpha=alpha, gamma=2.0)
    got = loss_fn(logits, labels)

    # hand-compute with the correct (1, C, 1, 1) broadcast
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_aligned = alpha.view(1, 3, 1, 1)
    alpha_t = alpha_aligned * labels + (1 - alpha_aligned) * (1 - labels)
    expected = (alpha_t * (1 - p_t) ** 2.0 * bce).mean()

    assert torch.allclose(got, expected)


def test_focal_loss_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    from src.models.distortion_head import StageBDistortionHead

    head = StageBDistortionHead(in_channels=8)
    features = torch.randn(6, 8, 4, 4)
    labels = torch.randint(0, 2, (6, 3, 4, 4)).float()
    loss_fn = build_stage_b_focal_loss()
    opt = torch.optim.SGD(head.parameters(), lr=1.0)

    loss_before = loss_fn(head(features), labels)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), labels)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()


# --- Localized SSD loss (supervisor request, Session 19+) ----------------


def test_ssd_loss_is_finite_on_random_inputs():
    loss_fn = build_stage_b_ssd_loss()
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 2, (2, 3, 4, 4)).float()
    assert torch.isfinite(loss_fn(logits, labels))


def test_ssd_loss_matches_hand_calculation():
    # Two samples, 1 class, 2x2 grid -- small enough to hand-compute.
    # logits all 0 -> sigmoid(0) = 0.5 for every tile, in both samples.
    # sample 0 targets [1,0,1,0]: each tile's sq diff = (0.5-1)^2 or (0.5-0)^2 = 0.25
    #   -> localized sum over the 4 tiles = 1.0
    # sample 1 targets [1,1,1,1]: every tile's sq diff = (0.5-1)^2 = 0.25 -> sum = 1.0
    # mean over the batch: (1.0 + 1.0) / 2 = 1.0
    logits = torch.zeros(2, 1, 2, 2)
    targets = torch.tensor([
        [[[1.0, 0.0], [1.0, 0.0]]],  # sample 0
        [[[1.0, 1.0], [1.0, 1.0]]],  # sample 1
    ])

    loss_fn = LocalizedSSDLoss()
    got = loss_fn(logits, targets)
    assert torch.allclose(got, torch.tensor(1.0))


def test_ssd_loss_zero_for_perfect_predictions():
    # logit=+inf isn't representable, but a very confident correct logit
    # drives sigmoid arbitrarily close to the target -> loss ~0.
    logits = torch.tensor([[[[20.0]]]])   # sigmoid(20) ~= 1.0
    targets = torch.tensor([[[[1.0]]]])
    loss_fn = LocalizedSSDLoss()
    assert loss_fn(logits, targets).item() < 1e-6


def test_ssd_loss_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    from src.models.distortion_head import StageBDistortionHead

    head = StageBDistortionHead(in_channels=8)
    features = torch.randn(6, 8, 4, 4)
    labels = torch.randint(0, 2, (6, 3, 4, 4)).float()
    loss_fn = build_stage_b_ssd_loss()
    opt = torch.optim.SGD(head.parameters(), lr=1.0)

    loss_before = loss_fn(head(features), labels)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), labels)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()


# --- Dice + BCE loss (Stage C, architecture.md §4) -----------------------


def test_dice_bce_loss_is_finite_on_random_inputs():
    loss_fn = build_stage_c_loss()
    logits = torch.randn(2, 3, 8, 8)
    targets = torch.rand(2, 3, 8, 8)  # continuous [0,1], Stage C's actual target dtype
    assert torch.isfinite(loss_fn(logits, targets))


def test_dice_bce_loss_near_zero_for_confident_correct_predictions():
    # Every pixel confidently predicted matching its (binary) target ->
    # both BCE and soft Dice should be near zero.
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])  # sigmoid ~= 1.0 / ~= 0.0
    loss_fn = DiceBCELoss()
    assert loss_fn(logits, targets).item() < 1e-4


def test_dice_bce_loss_degenerate_all_zero_target_does_not_nan():
    # A genuinely clean sample (or an inactive class): target is all-zero.
    # The `smooth` term must prevent a 0/0 division in the Dice component.
    targets = torch.zeros(1, 1, 4, 4)
    logits = torch.full((1, 1, 4, 4), -10.0)  # confidently predicts "no distortion" too
    loss_fn = DiceBCELoss()
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() < 0.1  # correct confident prediction on an all-negative target -> small loss


def test_dice_is_batch_level_so_empty_target_samples_do_not_dominate():
    # Sample 0: a perfectly predicted positive region. Sample 1: empty
    # target with a small residual 0.01 probability everywhere. Batch-level
    # Dice stays small; per-sample Dice would put sample 1 at ~1.
    targets = torch.zeros(2, 1, 64, 64)
    targets[0, 0, :32] = 1.0
    logits = torch.full((2, 1, 64, 64), float(torch.logit(torch.tensor(0.01))))
    logits[0, 0, :32] = 20.0
    dice_only = DiceBCELoss(bce_weight=0.0)(logits, targets)
    assert dice_only.item() < 0.05


def test_dice_bce_loss_weight_interpolates_between_bce_and_dice():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    targets = torch.rand(2, 3, 8, 8)

    only_bce = DiceBCELoss(bce_weight=1.0)(logits, targets)
    only_dice = DiceBCELoss(bce_weight=0.0)(logits, targets)
    half = DiceBCELoss(bce_weight=0.5)(logits, targets)

    assert torch.allclose(half, 0.5 * only_bce + 0.5 * only_dice, atol=1e-5)


def test_dice_bce_loss_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    from src.models.distortion_head import StageCDistortionHead

    head = StageCDistortionHead(in_channels=8)
    features = torch.randn(2, 8, 2, 2)
    targets = torch.rand(2, 3, 64, 64)
    loss_fn = build_stage_c_loss()
    opt = torch.optim.SGD(head.parameters(), lr=1.0)

    loss_before = loss_fn(head(features), targets)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), targets)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()
