from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.gate import DEFAULT_GATE_IMG_SIZE, apply_gate, collect_gate_probs, load_gate
from src.models.distortion_head import ImpairedGateHead


def test_apply_gate_zeroes_predictions_for_images_below_threshold():
    # 3 images, 2 classes, 2x2 grid -- images 0 and 2 are "not impaired"
    # (gate prob < 0.5), image 1 is "impaired" (gate prob >= 0.5).
    probs = np.ones((3, 2, 2, 2), dtype=np.float32) * 0.7
    gate_probs = np.array([0.1, 0.9, 0.3])

    gated = apply_gate(probs, gate_probs, threshold=0.5)

    assert np.all(gated[0] == 0.0)
    assert np.all(gated[1] == 0.7)
    assert np.all(gated[2] == 0.0)


def test_apply_gate_accepts_dict_keyed_by_index():
    probs = np.ones((2, 1, 1, 1), dtype=np.float32) * 0.9
    gate_probs = {0: 0.4, 1: 0.6}

    gated = apply_gate(probs, gate_probs, threshold=0.5)

    assert gated[0, 0, 0, 0] == 0.0
    assert gated[1, 0, 0, 0] == 0.9


def test_apply_gate_does_not_mutate_input_array():
    probs = np.ones((1, 1, 1, 1), dtype=np.float32)
    gate_probs = np.array([0.0])

    gated = apply_gate(probs, gate_probs, threshold=0.5)

    assert gated[0, 0, 0, 0] == 0.0
    assert probs[0, 0, 0, 0] == 1.0  # original untouched


def test_apply_gate_custom_threshold():
    probs = np.ones((2, 1, 1, 1), dtype=np.float32)
    gate_probs = np.array([0.6, 0.85])

    gated = apply_gate(probs, gate_probs, threshold=0.8)

    assert gated[0, 0, 0, 0] == 0.0  # 0.6 < 0.8 -> gated out
    assert gated[1, 0, 0, 0] == 1.0  # 0.85 >= 0.8 -> kept


class _RecordingBackbone(torch.nn.Module):
    """Stands in for a multi-scale FrozenYOLOBackbone: records input sizes,
    returns a list of two feature maps."""

    def __init__(self):
        super().__init__()
        self.seen_sizes = []

    def forward(self, x):
        self.seen_sizes.append(tuple(x.shape[-2:]))
        return [torch.zeros(x.shape[0], 2, 4, 4), torch.ones(x.shape[0], 8, 2, 2)]


def test_collect_gate_probs_resizes_to_gate_img_size():
    backbone = _RecordingBackbone()
    gate_head = ImpairedGateHead(in_channels=[2, 8])
    loader = [(torch.zeros(3, 3, 32, 32), None)]

    probs = collect_gate_probs(backbone, gate_head, loader, torch.device("cpu"), img_size=64)

    assert backbone.seen_sizes == [(64, 64)]
    assert sorted(probs) == [0, 1, 2]


_WEIGHTS = Path("weights/yolo11m.pt")


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="requires weights/yolo11m.pt")
def test_load_gate_builds_its_own_backbone_from_the_checkpoint(tmp_path):
    from src.eval.gate import build_gate

    _, p5_head = build_gate("p5", str(_WEIGHTS), torch.device("cpu"))
    old = tmp_path / "old.pt"
    torch.save({"head_state_dict": p5_head.state_dict()}, old)
    backbone, head, img_size = load_gate(old, str(_WEIGHTS), torch.device("cpu"))
    assert img_size == DEFAULT_GATE_IMG_SIZE == 640 and not head.multiscale

    _, ms_head = build_gate("multiscale", str(_WEIGHTS), torch.device("cpu"), taps=(2, 32))
    new = tmp_path / "new.pt"
    torch.save({"head_state_dict": ms_head.state_dict(), "arch": "multiscale", "taps": [2, 32],
                "img_size": 512}, new)
    backbone, head, img_size = load_gate(new, str(_WEIGHTS), torch.device("cpu"))
    assert img_size == 512 and head.multiscale
    assert len(backbone(torch.zeros(1, 3, 64, 64))) == 2
