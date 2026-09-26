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
import torch.nn.functional as F

from src.models.distortion_head import ImpairedGateHead

# Every impaired-gate checkpoint so far was trained at train_impaired_gate.py's
# default --img-size; newer checkpoints record their own "img_size".
DEFAULT_GATE_IMG_SIZE = 640


def load_gate(checkpoint, in_channels, device):
    """Returns (gate_head, gate_img_size) -- the head in eval mode, plus the
    input resolution it was trained at (pass to collect_gate_probs).
    `in_channels` is the backbone's out_channels the gate will be fed from:
    a P5-only gate (checkpoint arch "p5", the default for older ones) uses
    just P5 (the last entry of a multi-scale list); a "multiscale" gate needs
    the same multi-scale backbone it was trained on."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("arch", "p5") == "multiscale":
        if not isinstance(in_channels, (list, tuple)):
            raise ValueError("a multiscale gate needs a multi-scale backbone "
                             "(FrozenYOLOBackbone(return_layers=...)), got a P5-only one")
    elif isinstance(in_channels, (list, tuple)):
        in_channels = in_channels[-1]
    head = ImpairedGateHead(in_channels=in_channels).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return head, ckpt.get("img_size", DEFAULT_GATE_IMG_SIZE)


@torch.no_grad()
def collect_gate_probs(backbone, gate_head, loader, device, img_size=None):
    """Runs the impaired-gate head (ImpairedGateHead) over the given
    shuffle=False loader, returning {dataset_index: P(impaired)} keyed by
    position in iteration order -- valid only when `loader` was built with
    shuffle=False.

    `img_size`: the gate's own training resolution. Stage B/C loaders
    produce 512x512 images, but the gate was trained at 640x640 -- a frozen
    backbone's GAP features shift with input scale, so feeding the gate a
    different resolution than it was trained on silently shifts its
    calibration. Images are bilinearly resized to `img_size` first when it
    differs. A backbone returning a list of feature maps (the multi-scale
    one) is passed whole to a multiscale gate, and reduced to its last (P5)
    entry for a P5-only one."""
    probs = {}
    idx = 0
    for images, _ in loader:
        images = images.to(device)
        if img_size is not None and images.shape[-1] != img_size:
            images = F.interpolate(images, size=(img_size, img_size), mode="bilinear", align_corners=False)
        features = backbone(images)
        if isinstance(features, list) and not gate_head.multiscale:
            features = features[-1]
        batch_probs = torch.softmax(gate_head(features), dim=1)[:, 1].cpu().numpy()
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
