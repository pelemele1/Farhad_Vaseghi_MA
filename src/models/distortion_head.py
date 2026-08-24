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
