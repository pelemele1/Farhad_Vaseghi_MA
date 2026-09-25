"""
Stage A distortion head (architecture.md §2, §6): "Global average pooling
over the P5 feature map -> small FC head... Custom nn.Module on P5 (Stage
A): GAP + 2x FC + sigmoid, ~a few lines." Outputs one score per
`{dirt, water, scratch}` (multi-label, architecture.md §2's recommended
variant).

Returns raw logits, not post-sigmoid probabilities: architecture.md §4
specifies the loss as "BCE-with-logits", which expects logits and applies
the sigmoid internally for numerical stability -- applying a Sigmoid layer
here too would double-apply it under that loss. The "...FC + sigmoid"
output architecture.md describes is still exactly what this produces, just
with the sigmoid folded into the loss during training; call
`torch.sigmoid(head(features))` to get the per-class probabilities directly
(e.g. at inference).
"""
import torch.nn as nn


class StageADistortionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, class_names=("dirt", "water", "scratch")):
        super().__init__()
        self.class_names = tuple(class_names)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.class_names)),
        )

    def forward(self, features):
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)


class StageBDistortionHead(nn.Module):
    """Stage B distortion head (architecture.md §2): "The feature map is
    treated as a grid ... one classification output per tile." A 1x1 conv
    applied directly to the P5 feature map does exactly that -- no pooling,
    no resizing, "the feature map" itself *is* the grid, one output per
    spatial position. At img_size=512 the frozen backbone's P5 (stride 32)
    is 16x16, matching the grid size architecture.md uses as its own
    example.

    Returns raw logits, shape (B, num_classes, H, W) -- same
    BCE-with-logits-expects-logits reasoning as StageADistortionHead; call
    `torch.sigmoid(head(features))` for per-tile-per-class probabilities."""

    def __init__(self, in_channels, class_names=("dirt", "water", "scratch")):
        super().__init__()
        self.class_names = tuple(class_names)
        self.conv = nn.Conv2d(in_channels, len(self.class_names), kernel_size=1)

    def forward(self, features):
        return self.conv(features)


class ImpairedGateHead(nn.Module):
    """Image-level binary "is this image impaired at all" gate (Session 20).
    Same GAP + 2xFC architecture as StageADistortionHead, but a distinct
    class: its 2 logits are mutually exclusive (index 0 = not_impaired,
    index 1 = impaired), trained with nn.CrossEntropyLoss/softmax -- NOT
    sigmoid/BCE like every other head in this file. Kept as its own class
    (rather than reusing StageADistortionHead with class_names=
    ("not_impaired", "impaired")) so that different loss family is visible
    at the type level to a reader skimming this module.

    Returns raw 2-logit output, shape (B, 2). Call
    `torch.softmax(head(features), dim=1)` for class probabilities, or
    `.argmax(dim=1)` for the hard decision."""

    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.class_names = ("not_impaired", "impaired")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features):
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)
