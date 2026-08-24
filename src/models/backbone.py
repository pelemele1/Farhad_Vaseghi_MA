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


class FrozenYOLOBackbone(nn.Module):
    def __init__(self, model_name="weights/yolo11m.pt"):
        super().__init__()
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
            probe = torch.zeros(1, 3, 64, 64)
            self.out_channels = self.forward(probe).shape[1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, mode=True):
        # Always frozen (Option 1: no detection fine-tuning this pass) --
        # stay in eval() regardless of the containing model's mode, so
        # batchnorm running stats never drift while the distortion head
        # trains.
        return super().train(False)
