import torch
import torch.nn as nn

from src.models.distortion_head import (
    ImpairedGateHead,
    StageADistortionHead,
    StageBDistortionHead,
    StageCDistortionHead,
)


def test_forward_output_shape_matches_class_count():
    head = StageADistortionHead(in_channels=512)
    features = torch.randn(4, 512, 4, 4)
    out = head(features)
    assert out.shape == (4, 3)


def test_forward_is_agnostic_to_spatial_size():
    head = StageADistortionHead(in_channels=16)
    features = torch.randn(2, 16, 7, 9)
    out = head(features)
    assert out.shape == (2, 3)


def test_class_names_default_and_override():
    head = StageADistortionHead(in_channels=8)
    assert head.class_names == ("dirt", "water", "scratch")

    custom = StageADistortionHead(in_channels=8, class_names=("a", "b"))
    assert custom.class_names == ("a", "b")
    assert custom(torch.randn(1, 8, 2, 2)).shape == (1, 2)


def test_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    head = StageADistortionHead(in_channels=8)
    features = torch.randn(6, 8, 3, 3)
    labels = torch.randint(0, 2, (6, 3)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(head.parameters(), lr=0.1)

    loss_before = loss_fn(head(features), labels)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), labels)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()


# --- Impaired gate head (image-level binary) -----------------------------


def test_impaired_gate_head_forward_output_shape():
    head = ImpairedGateHead(in_channels=512)
    features = torch.randn(4, 512, 4, 4)
    out = head(features)
    assert out.shape == (4, 2)


def test_impaired_gate_head_class_names():
    head = ImpairedGateHead(in_channels=8)
    assert head.class_names == ("not_impaired", "impaired")


def test_impaired_gate_head_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    head = ImpairedGateHead(in_channels=8)
    features = torch.randn(6, 8, 3, 3)
    labels = torch.randint(0, 2, (6,), dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(head.parameters(), lr=0.1)

    loss_before = loss_fn(head(features), labels)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), labels)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()


# --- Stage B (tile-grid head) -------------------------------------------


def test_stage_b_forward_output_shape_is_per_tile():
    head = StageBDistortionHead(in_channels=512)
    features = torch.randn(4, 512, 16, 16)
    out = head(features)
    assert out.shape == (4, 3, 16, 16)  # one score per class per tile, not pooled


def test_stage_b_forward_tracks_feature_map_spatial_size():
    head = StageBDistortionHead(in_channels=16)
    out = head(torch.randn(2, 16, 7, 9))
    assert out.shape == (2, 3, 7, 9)


def test_stage_b_class_names_default_and_override():
    head = StageBDistortionHead(in_channels=8)
    assert head.class_names == ("dirt", "water", "scratch")

    custom = StageBDistortionHead(in_channels=8, class_names=("a", "b"))
    assert custom.class_names == ("a", "b")
    assert custom(torch.randn(1, 8, 4, 4)).shape == (1, 2, 4, 4)


def test_stage_b_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    head = StageBDistortionHead(in_channels=8)
    features = torch.randn(6, 8, 4, 4)
    labels = torch.randint(0, 2, (6, 3, 4, 4)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(head.parameters(), lr=0.1)

    loss_before = loss_fn(head(features), labels)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), labels)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()


# --- Stage C (pixel-level decoder head) ----------------------------------


def test_stage_c_forward_upsamples_p5_back_to_input_resolution():
    # P5 stride is 32 -- a 512x512 input gives a 16x16 P5 feature map, and
    # the head's 5 blocks of 2x upsampling (2**5 == 32) must land exactly
    # back on 512x512.
    head = StageCDistortionHead(in_channels=512)
    features = torch.randn(2, 512, 16, 16)
    out = head(features)
    assert out.shape == (2, 3, 512, 512)


def test_stage_c_forward_tracks_feature_map_spatial_size():
    head = StageCDistortionHead(in_channels=16)
    out = head(torch.randn(2, 16, 4, 4))
    assert out.shape == (2, 3, 128, 128)  # 4 * 32


def test_stage_c_class_names_default_and_override():
    head = StageCDistortionHead(in_channels=8)
    assert head.class_names == ("dirt", "water", "scratch")

    custom = StageCDistortionHead(in_channels=8, class_names=("a", "b"))
    assert custom.class_names == ("a", "b")
    assert custom(torch.randn(1, 8, 2, 2)).shape == (1, 2, 64, 64)


def test_stage_c_single_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    head = StageCDistortionHead(in_channels=8)
    features = torch.randn(2, 8, 2, 2)
    targets = torch.randint(0, 2, (2, 3, 64, 64)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(head.parameters(), lr=0.1)

    loss_before = loss_fn(head(features), targets)
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = loss_fn(head(features), targets)

    assert torch.isfinite(loss_before)
    assert loss_after.item() < loss_before.item()
