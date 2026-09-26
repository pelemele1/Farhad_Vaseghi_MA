import numpy as np
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
    """Stands in for FrozenYOLOBackbone: records input sizes, returns a
    list of feature maps (like the multi-scale Stage C backbone) whose last
    entry is 'P5'."""

    def __init__(self):
        super().__init__()
        self.seen_sizes = []

    def forward(self, x):
        self.seen_sizes.append(tuple(x.shape[-2:]))
        return [torch.zeros(x.shape[0], 2, 4, 4), torch.ones(x.shape[0], 8, 2, 2)]


def test_collect_gate_probs_resizes_to_gate_img_size_and_uses_p5():
    backbone = _RecordingBackbone()
    gate_head = ImpairedGateHead(in_channels=8)
    loader = [(torch.zeros(3, 3, 32, 32), None)]

    probs = collect_gate_probs(backbone, gate_head, loader, torch.device("cpu"), img_size=64)

    assert backbone.seen_sizes == [(64, 64)]
    assert sorted(probs) == [0, 1, 2]


def test_load_gate_defaults_to_640_for_checkpoints_without_img_size(tmp_path):
    head = ImpairedGateHead(in_channels=8)
    old = tmp_path / "old.pt"
    torch.save({"head_state_dict": head.state_dict()}, old)
    new = tmp_path / "new.pt"
    torch.save({"head_state_dict": head.state_dict(), "img_size": 512}, new)

    assert load_gate(old, 8, torch.device("cpu"))[1] == DEFAULT_GATE_IMG_SIZE == 640
    assert load_gate(new, [4, 8], torch.device("cpu"))[1] == 512


def test_multiscale_gate_receives_every_feature_map():
    backbone = _RecordingBackbone()
    gate_head = ImpairedGateHead(in_channels=[2, 8])
    loader = [(torch.zeros(2, 3, 32, 32), None)]
    probs = collect_gate_probs(backbone, gate_head, loader, torch.device("cpu"))
    assert sorted(probs) == [0, 1]


def test_load_gate_multiscale_rejects_p5_only_backbone(tmp_path):
    import pytest

    head = ImpairedGateHead(in_channels=[2, 8])
    path = tmp_path / "ms.pt"
    torch.save({"head_state_dict": head.state_dict(), "arch": "multiscale", "img_size": 512}, path)
    loaded, img_size = load_gate(path, [2, 8], torch.device("cpu"))
    assert loaded.multiscale and img_size == 512
    with pytest.raises(ValueError):
        load_gate(path, 8, torch.device("cpu"))
