"""
Real inference-time use of ImpairedGateHead's image-level "is this image
impaired at all" decision to suppress Stage B's own tile predictions on
images the gate calls "not impaired" (Session 20, Round 2). Originally the
gate was reporting-only (annotated the report but never touched
probs/labels), deliberately, to avoid coupling the two heads' error rates --
reversed after the concrete failure mode showed up in the 5-column report's
clean-image rows (see docs/development_log.md, "Round 2"). The coupling
tradeoff this reintroduces is real and documented there: a gate false
negative now silently suppresses genuine Stage B detections too.
"""
import numpy as np
import torch


@torch.no_grad()
def collect_gate_probs(backbone, gate_head, loader, device):
    """Runs the impaired-gate head (ImpairedGateHead) over the given
    shuffle=False loader, returning {dataset_index: P(impaired)} keyed by
    position in iteration order -- valid only when `loader` was built with
    shuffle=False."""
    probs = {}
    idx = 0
    for images, _ in loader:
        images = images.to(device)
        batch_probs = torch.softmax(gate_head(backbone(images)), dim=1)[:, 1].cpu().numpy()
        for value in batch_probs:
            probs[idx] = float(value)
            idx += 1
    return probs


def apply_gate(probs, gate_probs, threshold=0.5):
    """probs: (N, C, H, W) Stage B tile-probability array. gate_probs: a
    {dataset_index: P(impaired)} dict (as returned by collect_gate_probs) or
    an (N,) array, aligned with probs' first axis. Zeroes every tile
    prediction for each image the gate calls "not impaired"
    (gate_probs[i] < threshold) -- the gate's decision is per-image, so it
    silences all classes' tile grids together for that image, not per-class.
    Returns a new array; `probs` itself is not modified in place."""
    if isinstance(gate_probs, dict):
        gate_probs = np.array([gate_probs[i] for i in range(len(probs))])
    gated = probs.copy()
    gated[gate_probs < threshold] = 0.0
    return gated
