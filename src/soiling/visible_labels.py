"""
Visible-change ground truth (Session 27): one class per pixel -- clean, dirt,
water or scratch -- given by what is actually visible in the rendered image,
not by the generator's soft masks.

For every applied effect e, the image is re-rendered with e left out (all
other effects with the same seeds; effects are pixel-reproducible since the
perlin seeding fix in effects.py). The pixel-wise change between the final
image and that re-render is e's *visible contribution*: where a later layer
hides e (dirt under a thick water film), the change is ~0, where e is on top
or transparent layers let it show through, it is large. A pixel is labeled
with the effect of largest contribution if that contribution exceeds
MIN_PIXEL_CHANGE gray levels (the same "visible" cutoff as the label audit),
else clean -- except that a visible scratch (always the top layer) wins over
the layers below it. Small gaps, speckles and pinholes are then cleaned up.

For a single-effect image the re-render without e is the clean source, so the
label is simply "where the image visibly differs from the clean photo".
"""
import cv2 as cv
import numpy as np

from src.soiling.effects import MIN_PIXEL_CHANGE, add_dirt, add_scratch, add_water, apply_severity

EFFECT_NAMES = ("dirt", "water", "scratch")
CLASS_NAMES = ("clean",) + EFFECT_NAMES  # label value = index (0 = clean)
_EFFECTS = {"dirt": add_dirt, "water": add_water, "scratch": add_scratch}

# Smoothing of the change map before the cutoff (Gaussian sigma, pixels): light
# for scratches, so a 1-px line with a change of 40 levels stays above 10, a
# little stronger for dirt and water, whose textures flicker around the cutoff.
CHANGE_BLUR_SIGMA = {"dirt": 1.2, "water": 1.2, "scratch": 0.7}
# A pixel can only take an effect's class within this many pixels of where the
# effect's own mask is non-zero: excludes faint global side effects (e.g. the
# thin-water renderer's 20% film over the whole frame) while keeping the blur
# halo around a droplet or a dirt patch.
MASK_SUPPORT_MIN = 0.02
MASK_SUPPORT_DILATE = 7
# Clean-up at native resolution: gaps up to CLOSE_KERNEL px between fragments of
# one class are closed, regions smaller than MIN_COMPONENT_PIXELS are removed,
# and clean holes smaller than MAX_HOLE_PIXELS inside a distortion are filled.
CLOSE_KERNEL = 5
MIN_COMPONENT_PIXELS = 32
MAX_HOLE_PIXELS = 64


def render_chain(image, steps):
    """Applies `steps` [(effect name, seed, severity), ...] in order, exactly
    as the dataset builder does (each severity blend relative to the image just
    before that effect). Returns (image, {name: full-strength mask})."""
    out = image
    masks = {}
    for name, seed, severity in steps:
        effect_out, mask = _EFFECTS[name](out, seed=seed)
        out, _ = apply_severity(out, effect_out, mask, severity)
        masks[name] = mask
    return out, masks


def _change(a, b, sigma):
    change = np.abs(a.astype(np.float32) - b.astype(np.float32)).max(axis=2)
    if sigma > 0:
        change = cv.GaussianBlur(change, (0, 0), sigma)
    return change


def visible_contributions(image, final, steps, masks):
    """{name: (H, W) float32 visible change of that effect in `final`}, zero
    outside the effect's (dilated) mask support."""
    kernel = np.ones((MASK_SUPPORT_DILATE, MASK_SUPPORT_DILATE), np.uint8)
    out = {}
    for i, (name, _, _) in enumerate(steps):
        others = steps[:i] + steps[i + 1:]
        without, _ = render_chain(image, others) if others else (image, None)
        support = cv.dilate((masks[name] > MASK_SUPPORT_MIN).astype(np.uint8), kernel) > 0
        out[name] = np.where(support, _change(final, without, CHANGE_BLUR_SIGMA[name]), 0.0).astype(np.float32)
    return out


def clean_up_labels(labels):
    """Closes small gaps between fragments of one class (clean pixels only),
    removes distortion regions smaller than MIN_COMPONENT_PIXELS and fills
    clean holes smaller than MAX_HOLE_PIXELS with the most common class around
    them."""
    labels = labels.copy()
    kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    for value in range(1, len(CLASS_NAMES)):
        region = (labels == value).astype(np.uint8)
        if region.any():
            closed = cv.morphologyEx(region, cv.MORPH_CLOSE, kernel) > 0
            labels[closed & (labels == 0)] = value

    for value in range(1, len(CLASS_NAMES)):
        region = (labels == value).astype(np.uint8)
        if not region.any():
            continue
        _, comp, stats, _ = cv.connectedComponentsWithStats(region, connectivity=8)
        small = np.where(stats[1:, cv.CC_STAT_AREA] < MIN_COMPONENT_PIXELS)[0] + 1
        if len(small):
            labels[np.isin(comp, small)] = 0

    _, comp, stats, _ = cv.connectedComponentsWithStats((labels == 0).astype(np.uint8), connectivity=4)
    small = np.where(stats[1:, cv.CC_STAT_AREA] < MAX_HOLE_PIXELS)[0] + 1
    if len(small):
        counts = np.stack([cv.blur((labels == v).astype(np.float32), (9, 9))
                           for v in range(1, len(CLASS_NAMES))])
        holes = np.isin(comp, small) & (counts.max(axis=0) > 0)
        labels[holes] = counts.argmax(axis=0)[holes] + 1
    return labels


def visible_label_map(image, final, steps, masks):
    """(H, W) uint8 class map (0 clean, 1 dirt, 2 water, 3 scratch) of the
    visible distortion at every pixel of `final`."""
    labels = np.zeros(final.shape[:2], dtype=np.uint8)
    if not steps:
        return labels
    contrib = visible_contributions(image, final, steps, masks)
    stack = np.stack([contrib.get(name, np.zeros(labels.shape, np.float32)) for name in EFFECT_NAMES])
    best = stack.argmax(axis=0)
    visible = stack.max(axis=0) > MIN_PIXEL_CHANGE
    labels[visible] = best[visible] + 1
    # A scratch is the top layer and a thin line: wherever it visibly changes the
    # image it is what one sees there, even over a dirt patch that alone would
    # change the pixel more.
    scratch = EFFECT_NAMES.index("scratch")
    labels[stack[scratch] > MIN_PIXEL_CHANGE] = scratch + 1
    return clean_up_labels(labels)


def tile_class_grid(labels, grid_h, grid_w, thresholds):
    """(grid_h, grid_w) uint8 class per tile: each distortion class's coverage
    of the tile (area fraction) is compared with its own cutoff (`thresholds`
    by name -- a thin scratch needs far less area than a dirt patch); the tile
    takes the class that exceeds its cutoff by the largest factor, else clean."""
    ratios = []
    for value, name in enumerate(EFFECT_NAMES, start=1):
        coverage = cv.resize((labels == value).astype(np.float32), (grid_w, grid_h), interpolation=cv.INTER_AREA)
        ratios.append(coverage / thresholds[name])
    ratios = np.stack(ratios)
    grid = (ratios.argmax(axis=0) + 1).astype(np.uint8)
    grid[ratios.max(axis=0) < 1.0] = 0
    return grid


def class_pixel_counts(labels):
    return {name: int((labels == value).sum()) for value, name in enumerate(CLASS_NAMES)}
