"""
Frozen COCO-pretrained YOLO backbone (architecture.md §3 Option 1, §6):
"a separate small model that takes the frozen YOLO backbone features as
input -- no intervention in the YOLO training loop needed." Exposes the P5
feature map (the backbone's deepest single-scale stage, stride 32) for
Stage A's distortion head. No detection fine-tuning on MIO-TCD in this pass
-- every parameter is frozen and the module is permanently kept in eval().
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

# Ultralytics' YOLOv11 DetectionModel.model is a flat nn.Sequential of 24
# layers (indices 0-23). Traced for yolo11m: layers 0-10 (Conv/C3k2 stages,
# SPPF, C2PSA) are the shared CSPDarknet-style backbone, each taking input
# only from the immediately preceding layer (`layer.f == -1`); layer 11 is
# the neck's first Upsample, where the FPN/PAN top-down/bottom-up fusion
# begins. Layer 10 (C2PSA)'s output is therefore the last single-scale
# feature map before any multi-scale fusion -- i.e. P5.
BACKBONE_END = 10
# Last layer at each stride (yolo11m @512: 2 -> 256ch/128px, 4 -> 512ch/64px,
# 6 -> 512ch/32px, 10 -> 512ch/16px) -- the skip taps for Stage C's decoder.
STRIDE_TAPS = {4: 2, 8: 4, 16: 6, 32: BACKBONE_END}


class FrozenYOLOBackbone(nn.Module):
    """forward() returns the P5 tensor by default. With `return_layers`
    (layer indices, e.g. STRIDE_TAPS values), returns a list of those layers'
    outputs in the given order instead, and `out_channels` is a matching
    list."""

    def __init__(self, model_name="weights/yolo11m.pt", return_layers=None):
        super().__init__()
        self.return_layers = tuple(return_layers) if return_layers is not None else None
        full_model = YOLO(model_name).model
        layers = full_model.model[: BACKBONE_END + 1]
        for layer in layers:
            assert layer.f == -1, (
                f"backbone layer {layer.__class__.__name__} (index {layer.i}) "
                f"takes input from {layer.f}, not just the previous layer -- "
                "this model's backbone isn't purely sequential, BACKBONE_END "
                "or this forward pass needs updating"
            )
        self.layers = layers

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        with torch.no_grad():
            probe = self.forward(torch.zeros(1, 3, 64, 64))
            if self.return_layers is None:
                self.out_channels = probe.shape[1]
            else:
                self.out_channels = [f.shape[1] for f in probe]

    def forward(self, x):
        if self.return_layers is None:
            for layer in self.layers:
                x = layer(x)
            return x
        taps = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.return_layers:
                taps[i] = x
        return [taps[i] for i in self.return_layers]

    def train(self, mode=True):
        # Always frozen (Option 1: no detection fine-tuning this pass) --
        # stay in eval() regardless of the containing model's mode, so
        # batchnorm running stats never drift while the distortion head
        # trains.
        return super().train(False)
