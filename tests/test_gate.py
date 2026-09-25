import numpy as np

from src.eval.gate import apply_gate


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
