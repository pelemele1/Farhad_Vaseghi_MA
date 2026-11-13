from pathlib import Path

import pytest
import torch

from src.models.backbone import STRIDE_TAPS, FrozenYOLOBackbone

_WEIGHTS = Path("weights/yolo11m.pt")

pytestmark = pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason="requires weights/yolo11m.pt (COCO-pretrained YOLOv11m) downloaded locally",
)


def test_forward_produces_the_p5_feature_map():
    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    x = torch.zeros(2, 3, 128, 128)
    out = backbone(x)
    assert out.shape == (2, backbone.out_channels, 4, 4)  # stride 32


def test_all_parameters_are_frozen():
    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    assert all(not p.requires_grad for p in backbone.parameters())


def test_stays_in_eval_mode_even_after_train_call():
    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    backbone.train()
    assert not backbone.training
    assert all(not m.training for m in backbone.modules())


def test_return_layers_gives_one_map_per_stride_and_p5_matches_default():
    default = FrozenYOLOBackbone(str(_WEIGHTS))
    multi = FrozenYOLOBackbone(str(_WEIGHTS), return_layers=[STRIDE_TAPS[s] for s in (4, 8, 16, 32)])
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        feats = multi(x)
        p5 = default(x)

    assert [f.shape[-1] for f in feats] == [16, 8, 4, 2]
    assert multi.out_channels == [f.shape[1] for f in feats]
    assert multi.out_channels[-1] == default.out_channels
    assert torch.allclose(feats[-1], p5)
