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
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class StageCDistortionHead(nn.Module):
    """Stage C distortion head (architecture.md §2): "Small FCN/UNet-style
    decoder -> per-pixel classes." The frozen backbone (src/models/
    backbone.py) only exposes a single scale, P5 (stride 32) -- no skip
    connections from earlier, higher-resolution layers are available without
    changing the backbone wrapper, so v1 is a plain FCN: 5 blocks of
    (bilinear 2x upsample -> 3x3 conv -> ReLU), each halving the stride,
    which composes to exactly 32x (2**5) -- i.e. P5's spatial size times 32
    lands back on the exact input resolution the backbone was fed, so this
    head's output is (B, num_classes, img_size, img_size), matching its
    input pixel-for-pixel. Bilinear+conv rather than ConvTranspose2d avoids
    checkerboard upsampling artifacts and keeps the parameter count small
    (this project's established "start minimal" convention -- see Stage
    A/B's own heads above). A skip connection from an earlier backbone layer
    is a natural follow-up if segmentation quality proves inadequate, not
    something v1 needs to anticipate.

    Returns raw logits, shape (B, num_classes, H, W) -- same
    BCE-with-logits-expects-logits reasoning as the other heads in this
    file; call `torch.sigmoid(head(features))` for per-pixel-per-class
    probabilities."""

    def __init__(self, in_channels, class_names=("dirt", "water", "scratch"),
                 hidden_dim=64, n_upsample_blocks=5):
        super().__init__()
        self.class_names = tuple(class_names)
        blocks = []
        ch = in_channels
        for _ in range(n_upsample_blocks):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
            ch = hidden_dim
        self.decoder = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(hidden_dim, len(self.class_names), kernel_size=1)

    def forward(self, features):
        return self.out_conv(self.decoder(features))


def _conv_bn_relu(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class StageCUNetHead(nn.Module):
    """Stage C v2: the "UNet-style decoder" option architecture.md §2 names,
    using skip connections from the frozen backbone's stride-4/8/16 layers
    (FrozenYOLOBackbone(return_layers=STRIDE_TAPS values)) alongside P5. The
    v1 FCN (StageCDistortionHead) must reconstruct every edge from P5's 16x16
    grid alone -- hopeless for 1-7px scratches; the earlier layers still carry
    that fine spatial detail.

    features: list ordered fine -> coarse, [s4, s8, s16, s32]. Each is
    reduced to `hidden_dim` channels by a 1x1 lateral conv; the decoder walks
    coarse -> fine (upsample 2x, concat the lateral skip, 3x3 conv-BN-ReLU),
    predicts logits at stride 4, and bilinearly upsamples those logits 4x to
    the input resolution (cheaper than running full-resolution convs, and
    stride 4 is already far finer than the 64x64 evaluation grid).

    Returns raw logits, shape (B, num_classes, 4*H_s4, 4*W_s4)."""

    def __init__(self, in_channels_list, class_names=("dirt", "water", "scratch"), hidden_dim=64):
        super().__init__()
        self.class_names = tuple(class_names)
        self.laterals = nn.ModuleList(_conv_bn_relu(c, hidden_dim, kernel_size=1) for c in in_channels_list)
        self.fuse = nn.ModuleList(
            _conv_bn_relu(2 * hidden_dim, hidden_dim) for _ in range(len(in_channels_list) - 1)
        )
        self.refine = _conv_bn_relu(hidden_dim, hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, len(self.class_names), kernel_size=1)

    def forward(self, features):
        lats = [lat(f) for lat, f in zip(self.laterals, features)]
        x = lats[-1]
        for fuse, skip in zip(self.fuse, reversed(lats[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = fuse(torch.cat([x, skip], dim=1))
        logits = self.out_conv(self.refine(x))
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)


class StageBMultiScaleHead(nn.Module):
    """Stage B v2: still one classification output per P5 grid tile
    (architecture.md §2), but each tile sees features from the frozen
    backbone's stride-4/8/16 layers too, area-pooled down to the tile grid,
    not just P5 -- the same finer features that lifted Stage C's faint-
    distortion AP roughly 2x (docs/development_log.md Session 22). The v1
    head (StageBDistortionHead, one 1x1 conv on P5) only sees the coarsest,
    most semantic layer, where faint texture changes are largely gone.

    features: list ordered fine -> coarse, [s4, s8, s16, s32]. Returns raw
    logits, shape (B, num_classes, H_s32, W_s32)."""

    def __init__(self, in_channels_list, class_names=("dirt", "water", "scratch"), hidden_dim=64):
        super().__init__()
        self.class_names = tuple(class_names)
        self.laterals = nn.ModuleList(_conv_bn_relu(c, hidden_dim, kernel_size=1) for c in in_channels_list)
        self.fuse = _conv_bn_relu(hidden_dim * len(in_channels_list), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, len(self.class_names), kernel_size=1)

    def forward(self, features):
        grid = features[-1].shape[-2:]
        pooled = [F.adaptive_avg_pool2d(lat(f), grid) for lat, f in zip(self.laterals, features)]
        return self.out_conv(self.fuse(torch.cat(pooled, dim=1)))


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
    `.argmax(dim=1)` for the hard decision.

    `in_channels` may be a list (a multi-scale backbone's out_channels): each
    feature map is then average-pooled separately and the results
    concatenated, so the gate also sees the finer layers' texture cues
    (Session 22) -- with P5 alone it let ~48% of clean images through at a
    threshold keeping 95% of impaired ones."""

    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.class_names = ("not_impaired", "impaired")
        self.multiscale = isinstance(in_channels, (list, tuple))
        total = sum(in_channels) if self.multiscale else in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features):
        if self.multiscale:
            pooled = torch.cat([self.pool(f).flatten(1) for f in features], dim=1)
        else:
            pooled = self.pool(features).flatten(1)
        return self.fc(pooled)
