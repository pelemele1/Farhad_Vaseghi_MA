"""
Rasterizes a per-pixel distortion mask (as returned by `add_dirt`/`add_water`/
`add_scratch` in effects.py) down to a coarse per-tile binary label for
Stage B (architecture.md §2): "The feature map is treated as a grid (e.g.
16x16), one classification output per tile."

Must be called on the *same* mask returned alongside the image it labels --
`add_dirt`/`add_water` are only label-reproducible across separate calls with
the same seed, not pixel-reproducible (see docs/development_log.md Session 6:
a pythonperlin dependency silently reseeds np.random internally), so
recomputing a mask after the fact from a saved seed would not reliably line
up with an already-saved image's actual pixels.
"""
import cv2 as cv
import numpy as np


def rasterize_tile_label(mask, grid_h, grid_w, threshold):
    """mask: float32 [0, 1] array (H, W), as returned by an effect function.
    Downsamples via area-average pooling to (grid_h, grid_w) -- each output
    cell is the mean mask value ("coverage fraction") over the pixels that
    fall in that tile -- then thresholds to a binary uint8 label: a tile
    counts as positive if its coverage fraction is >= `threshold`.

    cv.resize(..., interpolation=INTER_AREA) is used for the downsampling
    step: it's an area-weighted average of the source pixels covered by each
    destination pixel, which is exactly "mean coverage fraction per tile"
    when downsampling (not just a generic resize)."""
    mask = np.asarray(mask, dtype=np.float32)
    coverage = cv.resize(mask, (grid_w, grid_h), interpolation=cv.INTER_AREA)
    return (coverage >= threshold).astype(np.uint8)
