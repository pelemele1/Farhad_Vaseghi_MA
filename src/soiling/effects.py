"""
Synthetic distortion effects for Stage A (architecture.md §2): dirt, water,
scratch. Each effect builds a per-pixel [0, 1] mask and alpha-blends it onto
the source image, so effects can be composed (multiple effects applied to
the same image for multi-label training) and the mask itself is available
as an intermediate artifact if needed later (e.g. for Stage C).

`add_scratch` is a custom procedural generator -- physical_lens_soiling
(github.com/JannLi/physical_lens_soiling, the repo overview.md names as the
synthetic-data source) has no scratch effect, and no ready-made lens-scratch
codebase was found that transfers cleanly (FilmDamageSimulator is scanned
35mm film grain damage; ScratchSim's public code is a BlenderProc 3D
pipeline for a rendered toy car, not usable on flat photos). Chosen after
comparing renders from both against this generator on a sample image -- see
docs/development_log.md for the comparison and rationale.
"""
import sys
from pathlib import Path

import numpy as np
import cv2 as cv

_THIRD_PARTY_DIR = Path(__file__).resolve().parents[2] / "third_party" / "physical_lens_soiling"
if str(_THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_DIR))

from add_mud import (  # noqa: E402 (path must be set up first)
    add_mudByTxture,
    add_dirtwaterByTxture,
    add_dirtwaterByTxture_slight,
)
from add_droplet_distort import add_distort  # noqa: E402
from generate_texture_paper import generate_texture  # noqa: E402

# Measured empirically (mean mask value / fraction of pixels > 50 out of 255,
# 3 trials each, unchanged vendored code): r_fog/thick_fog/little_rain_drop
# ~51% coverage, f_water_mud/big_rain_drop ~40%, but r_water_mud/many_rain_drop
# only ~9-13% and many_dust_drop ~2% -- those three read as near-invisible
# despite "r_water_mud" sounding like the canonical mud texture. Restricted
# to the modes that reliably give paper-Figure-1-strength coverage.
_DIRT_WATER_TEXTURE_MODS = [
    "r_fog", "thick_fog", "f_water_mud",
    "big_rain_drop", "little_rain_drop",
]


def _smooth_curve(n_waypoints, length, angle, jitter, n_samples=300):
    """Waypoints along a roughly straight direction with lateral jitter,
    upsampled to a dense polyline via per-axis linear interpolation."""
    t_way = np.linspace(0, 1, n_waypoints)
    along = t_way * length
    lateral = np.cumsum(np.random.uniform(-jitter, jitter, size=n_waypoints))
    lateral -= lateral[0]

    t_dense = np.linspace(0, 1, n_samples)
    along_dense = np.interp(t_dense, t_way, along)
    lateral_dense = np.interp(t_dense, t_way, lateral)

    ca, sa = np.cos(angle), np.sin(angle)
    dx = along_dense * ca - lateral_dense * sa
    dy = along_dense * sa + lateral_dense * ca
    return np.stack([dx, dy], axis=1)


def _visibility_mask(n, n_gaps=(2, 5), gap_frac=(0.05, 0.18)):
    """A real scratch on a lens only catches light -- and so only becomes
    visible -- at certain points along its length; the rest reads as a thin
    gap (confirmed against a real scratched-lens reference photo during the
    method comparison). Model this as random 'off' runs cut into an
    otherwise 'on' signal, so the rendered streak is broken/dashed rather
    than one unbroken line."""
    vis = np.ones(n, dtype=bool)
    for _ in range(np.random.randint(*n_gaps)):
        gap_len = int(np.random.uniform(*gap_frac) * n)
        if gap_len < 1:
            continue
        start = np.random.randint(0, max(1, n - gap_len))
        vis[start : start + gap_len] = False
    return vis


def _meandering_width(n, base_width, n_lobes=(2, 4)):
    """Width wanders between thick and thin along the streak instead of a
    single smooth taper -- real scratches vary irregularly in depth along
    their length, not just fading at the tips."""
    lobes = np.random.randint(*n_lobes)
    signal = np.zeros(n)
    for _ in range(lobes):
        center = np.random.uniform(0, n)
        spread = np.random.uniform(n * 0.08, n * 0.25)
        amp = np.random.uniform(0.4, 1.0)
        signal += amp * np.exp(-0.5 * ((np.arange(n) - center) / spread) ** 2)
    signal = signal / (signal.max() + 1e-6)
    end_taper = np.sin(np.linspace(0, np.pi, n)) ** 0.4  # still thin right at the very tips
    width_scale = (0.25 + 0.75 * signal) * end_taper
    return np.clip(base_width * width_scale, 0.6, None)


def _draw_tapered_streak(mask, curve_xy, origin, base_width, brightness):
    n = len(curve_xy)
    width_profile = _meandering_width(n, base_width)
    jitter = 1.0 + 0.15 * np.random.randn(n)  # brightness flicker along the length
    visible = _visibility_mask(n)
    layer = np.zeros_like(mask, dtype=np.float32)
    for i in range(n - 1):
        if not (visible[i] and visible[i + 1]):
            continue
        p1 = tuple((origin + curve_xy[i]).astype(int))
        p2 = tuple((origin + curve_xy[i + 1]).astype(int))
        w = max(1, int(round(width_profile[i])))
        val = float(np.clip(brightness * jitter[i], 0, 1))
        cv.line(layer, p1, p2, val, thickness=w, lineType=cv.LINE_AA)
    np.maximum(mask, layer, out=mask)


def generate_scratch_mask(shape, n_scratches=(2, 5), length_frac=(0.15, 0.55),
                           width_px=(1, 7), opacity=(0.6, 1.0), blur_sigma=0.6,
                           seed=None):
    """Returns a float32 [0, 1] mask, 0 = clean, higher = more distortion."""
    if seed is not None:
        np.random.seed(seed)
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    diag = float(np.hypot(H, W))
    num = np.random.randint(n_scratches[0], n_scratches[1] + 1)

    for _ in range(num):
        length = np.random.uniform(*length_frac) * diag
        angle = np.random.uniform(0, 2 * np.pi)
        origin = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
        curve = _smooth_curve(
            n_waypoints=np.random.randint(4, 7),
            length=length,
            angle=angle,
            jitter=length * 0.03,
        )
        base_width = np.random.uniform(*width_px)
        brightness = np.random.uniform(*opacity)
        _draw_tapered_streak(mask, curve, origin, base_width, brightness)

    if blur_sigma > 0:
        mask = cv.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    return np.clip(mask, 0, 1)


def add_scratch(image, **kwargs):
    """Alpha-blends procedural scratches onto `image` as bright refraction
    highlights (screen blend). Returns (distorted_image, mask)."""
    H, W = image.shape[:2]
    mask = generate_scratch_mask((H, W), **kwargs)
    alpha3 = cv.merge([mask, mask, mask])
    img_f = image.astype(np.float32) / 255.0
    screen = 1 - (1 - img_f) * (1 - alpha3)
    out = img_f * (1 - alpha3 * 0.97) + screen * (alpha3 * 0.97)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out, mask


def _normalize_mask(mask):
    """Vendored physical_lens_soiling masks come back uint8 0-255, sometimes
    with a trailing channel dim -- collapse to a single-channel float32
    [0, 1] mask, matching add_scratch's contract."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    return np.clip(mask.astype(np.float32) / 255.0, 0, 1)


def _random_dirt_water_texture():
    """physical_lens_soiling's dirt/water functions take a procedurally
    generated texture as an argument (not self-contained) -- pick one of
    their texture styles at random, as their own top-level demo code does."""
    mod = np.random.choice(_DIRT_WATER_TEXTURE_MODS)
    texture = generate_texture(mod=mod)
    if isinstance(texture, tuple):
        texture = texture[0]
    if texture.ndim == 2:
        texture = texture[:, :, np.newaxis]
    return texture


def add_dirt(image, seed=None):
    """Ported from physical_lens_soiling's `add_mudByTxture` (mud -> our
    `dirt`) -- see third_party/physical_lens_soiling/NOTICE.md. Returns
    (distorted_image, mask)."""
    if seed is not None:
        np.random.seed(seed)
    texture = _random_dirt_water_texture()
    out, mask = add_mudByTxture(image.copy(), texture)
    return out, _normalize_mask(mask)


def add_water(image, seed=None, mechanism=None):
    """Ported from physical_lens_soiling. `water` has two physically
    different mechanisms in the source repo, picked at random unless
    `mechanism` is given explicitly:
      - 'thick' / 'thin': a translucent blob blended onto the image
        (`add_dirtwaterByTxture` / `_slight`), same texture-driven style as
        `add_dirt`, just a lighter/whiter color.
      - 'droplet': an actual optical refraction/warp (`add_distort`) --
        radial "lens bulge" distortion, closer to how a real water droplet
        bends light rather than just tinting a region.
    Returns (distorted_image, mask). See
    third_party/physical_lens_soiling/NOTICE.md."""
    if seed is not None:
        np.random.seed(seed)
    if mechanism is None:
        mechanism = np.random.choice(["thick", "thin", "droplet"])

    if mechanism == "droplet":
        out, mask = add_distort(image.copy())
    else:
        texture = _random_dirt_water_texture()
        fn = add_dirtwaterByTxture if mechanism == "thick" else add_dirtwaterByTxture_slight
        out, mask = fn(image.copy(), texture)

    return out, _normalize_mask(mask)
