import torch
import torch.nn as nn

from src.models.distortion_head import StageADistortionHead


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
