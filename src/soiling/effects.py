"""
Synthetic contamination effects.

The six approved generators (Session 29, docs/synthetic_generators.md), one per
contamination type of the capture protocol, all reached through
`render_effect(name, image, seed, severity)`:
    water            add_water_droplets     Zhang et al. 2025 droplets (Fig. 1 g)
    dirt  (mud)      add_mud_stains         vendored add_mudByTxture (Fig. 1 b, c)
    fogging          add_fog                vendored mist shapes, our scattering veil
    cleaning_streaks add_cleaning_streaks   our own design
    scratch          add_sanding_scratches  sandpaper patches, this module's line drawer
    dust             add_dust               Fig. 1 e, vendored dust texture
Severity ("low" / "medium" / "high") is the amount of contamination, and a lighter
level is contained in the heavier one for the same seed.

The earlier effects -- `add_dirt`, `add_water`, `add_scratch`, faded by
`apply_severity` -- built the current 3-class dataset and stay for the
diagnostics scripts and tests. Each effect returns (image, per-pixel [0, 1] mask).

`add_scratch` is a custom procedural generator -- physical_lens_soiling
(github.com/JannLi/physical_lens_soiling, the repo overview.md names as the
synthetic-data source) has no scratch effect, and no ready-made lens-scratch
codebase was found that transfers cleanly (FilmDamageSimulator is scanned
35mm film grain damage; ScratchSim's public code is a BlenderProc 3D
pipeline for a rendered toy car, not usable on flat photos). Chosen after
comparing renders from both against this generator on a sample image -- see
docs/development_log.md for the comparison and rationale.
"""
import random
import sys
from pathlib import Path

import numpy as np
import cv2 as cv


def _seed_all(seed):
    """Some vendored functions (add_distort, via `random.choice`) use
    Python's stdlib `random` module, not `np.random` -- seeding only numpy
    leaves that draw dependent on whatever global stdlib random state
    happens to be left over from earlier calls in the same process, which
    breaks reproducibility across repeated builds. Seed both."""
    np.random.seed(seed)
    random.seed(seed)

_THIRD_PARTY_DIR = Path(__file__).resolve().parents[2] / "third_party" / "physical_lens_soiling"
if str(_THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_DIR))

from add_mud import (  # noqa: E402 (path must be set up first)
    add_mudByTxture,
    add_dirtwaterByTxture,
    add_dirtwaterByTxture_slight,
)
from add_droplet_distort import (  # noqa: E402
    add_distort,
    droplet_distort_onedirection,
    generate_distort_enlarege,
    rand_point,
)
import generate_texture_paper as _texture_module  # noqa: E402
from generate_texture_paper import generate_texture  # noqa: E402

# pythonperlin's perlin(seed=None) calls np.random.seed(None), i.e. reseeds numpy
# from OS entropy on every call -- the reason dirt/water renders were never
# pixel-reproducible (Session 6). Passing a seed drawn from the already-seeded
# numpy state makes every texture, and every draw after it, reproducible
# (Session 27: the visible-change labels re-render a combo without one effect).
_unseeded_perlin = _texture_module.perlin


def _seeded_perlin(*shape, seed=None, **kwargs):
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    return _unseeded_perlin(*shape, seed=seed, **kwargs)


_texture_module.perlin = _seeded_perlin

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
        _seed_all(seed)
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


def add_scratch(image, seed=None):
    """Alpha-blends procedural scratches onto `image` as bright refraction
    highlights (screen blend). Amount/severity is randomized internally
    (like `add_dirt`/`add_water`, not exposed as call-time parameters) --
    use `generate_scratch_mask` directly for explicit control over
    count/width/opacity/length. Returns (distorted_image, mask)."""
    H, W = image.shape[:2]
    mask = generate_scratch_mask((H, W), seed=seed)
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
        _seed_all(seed)
    texture = _random_dirt_water_texture()
    out, mask = add_mudByTxture(image.copy(), texture)
    return out, _normalize_mask(mask)


SEVERITY_LEVELS = ("low", "medium", "high")
SEVERITY_ALPHA = {"low": 0.3, "medium": 0.6, "high": 1.0}


def apply_severity(image, out, mask, severity):
    """Blends a full-strength effect output (`out`, `mask`, as returned by
    add_dirt/add_water/add_scratch) back toward the original clean `image`
    by `SEVERITY_ALPHA[severity]`, scaling `mask` by the same factor (the
    returned mask is the effective rendered opacity). The dataset builder
    deliberately does NOT use this scaled mask as ground truth -- see
    `src.soiling.dataset_builder.apply_effect_combo_with_masks`. `severity
    == "high"` (alpha=1.0) returns `out`/`mask` unchanged -- identical to
    today's unparametrized full-strength effect."""
    alpha = SEVERITY_ALPHA[severity]
    if alpha >= 1.0:
        return out, mask
    image_f = image.astype(np.float32)
    out_f = out.astype(np.float32)
    blended = image_f + alpha * (out_f - image_f)
    return np.clip(blended, 0, 255).astype(np.uint8), mask * alpha


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
        _seed_all(seed)
    if mechanism is None:
        mechanism = np.random.choice(["thick", "thin", "droplet"])

    if mechanism == "droplet":
        out, mask = add_distort(image.copy())
    else:
        texture = _random_dirt_water_texture()
        fn = add_dirtwaterByTxture if mechanism == "thick" else add_dirtwaterByTxture_slight
        out, mask = fn(image.copy(), texture)

    mask = _normalize_mask(mask)
    if mechanism == "thin":
        mask = remove_thin_water_floor(mask)
    return out, mask


# The approved water class: only the water droplets of Zhang et al. 2025, Fig. 1(g).
# Severity changes how much of the image the droplets cover, not just how faint they are:
# severity -> (droplet count range, droplet scale range, strength).
WATER_DROPLET_SEVERITY = {
    "low": ((2, 3), (0.30, 0.45), 0.90),
    "medium": ((4, 5), (0.40, 0.60), 0.95),
    "high": ((6, 9), (0.40, 0.83), 1.00),
}
# The vendored blur sizes are fixed in pixels and suit images much larger than ours.
DROPLET_WORK_WIDTH = 1440


def add_water_droplets(image, severity, seed=None):
    """The vendored droplet functions, called as `generate_manydroplet_dis` calls them,
    with the droplets set by `severity`. Only the droplets' change is carried back
    from the enlarged image, so pixels outside them stay exactly as they were.
    Returns (distorted_image, full-strength mask)."""
    if seed is not None:
        _seed_all(seed)
    (fewest, most), scale_range, strength = WATER_DROPLET_SEVERITY[severity]
    H, W = image.shape[:2]
    big = cv.resize(image, (DROPLET_WORK_WIDTH, round(H * DROPLET_WORK_WIDTH / W)), interpolation=cv.INTER_CUBIC)
    mode = random.choice(["enlarge", "one_direction", "mix_direction"])
    count = int(np.random.randint(fewest, most + 1))
    points = rand_point(n=3, if_random=1)[:count]
    scales = np.random.uniform(*scale_range, count)
    out = big.copy()
    mask = np.zeros(big.shape[:2], np.float32)
    for i in range(count):
        one_way = mode == "one_direction" or (mode == "mix_direction" and i % 2 == 1)
        render = droplet_distort_onedirection if one_way else generate_distort_enlarege
        out, droplet = render(out.copy(), shape=big.shape, center=points[i], scale=(scales[i], scales[i]),
                              random_seed=seed)
        np.maximum(mask, _normalize_mask(droplet), out=mask)
    shrink = (W, H)
    delta = (cv.resize(out, shrink, interpolation=cv.INTER_AREA).astype(np.float32)
             - cv.resize(big, shrink, interpolation=cv.INTER_AREA).astype(np.float32))
    result = np.clip(image.astype(np.float32) + strength * delta, 0, 255).astype(np.uint8)
    return result, cv.resize(mask, shrink, interpolation=cv.INTER_AREA)


# The approved mud class: only the mud stains of Zhang et al. 2025, Fig. 1(b, c), drawn by
# the vendored add_mudByTxture. Our names for the vendored stain shapes (texture modes):
MUD_SHAPES = {
    "scattered_spots": "f_water_mud",  # many separate, medium-sized spots
    "large_splashes": "big_rain_drop",  # one or a few large, irregular splashes
    "single_stain": "little_rain_drop",  # one large central stain with a grey fringe
}
# Share of the heavy stain area kept at each severity: lower severities shrink every
# stain and let the smallest vanish, so the light stains are also in medium and heavy.
MUD_KEEP = {"low": 0.20, "medium": 0.50, "high": 1.0}


def _shrink_stains(texture, keep):
    """Lowers a [0, 1] texture until its opaque part (add_mudByTxture's mask > 0.5)
    is `keep` of the original opaque area."""
    full = (texture > 0.5).mean()
    low, high = 0.0, 1.0
    for _ in range(30):
        drop = (low + high) / 2
        area = (np.clip(texture - drop, 0, None) / (1 - drop) > 0.5).mean()
        low, high = (drop, high) if area > keep * full else (low, drop)
    return np.clip(texture - drop, 0, None)


def add_mud_stains(image, severity, seed=None, shape=None):
    """Mud stains on `image` at `severity`; `shape` is one of MUD_SHAPES (random if None).
    Rendered on the image enlarged to DROPLET_WORK_WIDTH, like the droplets, and only
    the mud's change is carried back. Returns (distorted_image, full-strength mask)."""
    if seed is not None:
        _seed_all(seed)
    shape = shape or str(np.random.choice(sorted(MUD_SHAPES)))
    texture = generate_texture(mod=MUD_SHAPES[shape])
    texture = np.asarray(texture[0] if isinstance(texture, tuple) else texture, dtype=np.float32)
    texture = texture.max(axis=2) if texture.ndim == 3 else texture
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
    if MUD_KEEP[severity] < 1:
        texture = _shrink_stains(texture, MUD_KEEP[severity])
    H, W = image.shape[:2]
    big = cv.resize(image, (DROPLET_WORK_WIDTH, round(H * DROPLET_WORK_WIDTH / W)), interpolation=cv.INTER_CUBIC)
    out, mask = add_mudByTxture(big.copy(), texture[:, :, np.newaxis])
    shrink = (W, H)
    delta = (cv.resize(out, shrink, interpolation=cv.INTER_AREA).astype(np.float32)
             - cv.resize(big, shrink, interpolation=cv.INTER_AREA).astype(np.float32))
    result = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return result, cv.resize(_normalize_mask(mask), shrink, interpolation=cv.INTER_AREA)


# The approved fogging class: the mist shapes of Zhang et al. 2025, Fig. 1(f) (the vendored
# mist textures), heavily smoothed into one continuous veil. Inside the veil the scene loses
# sharpness and contrast and takes on its own spread-out light, a little brightened -- how
# condensation scatters light. This rendering is ours; the repo has no mist function.
FOG_SHAPES = {"patchy_mist": "r_fog", "dense_mist": "thick_fog"}
# severity -> (share of the fogged area kept, density, veil start in the texture, blur sigma)
FOG_SEVERITY = {
    "low": (0.40, 0.45, 0.20, 6),
    "medium": (0.70, 0.75, 0.12, 9),
    "high": (1.00, 1.00, 0.05, 12),
}
FOG_EDGE = 0.25  # texture value counted as fogged when shrinking the veil
FOG_BRIGHTEN = 0.45  # how far the spread-out light is lifted toward white


def _shrink_veil(texture, keep, edge=FOG_EDGE):
    """Lowers a smooth [0, 1] texture until its area above `edge` is `keep` of the
    original, then stretches it back to [0, 1]: the veil shrinks back from its edges."""
    full = (texture > edge).mean()
    low, high = 0.0, 1.0
    for _ in range(30):
        drop = (low + high) / 2
        area = (np.clip(texture - drop, 0, None) / (1 - drop) > edge).mean()
        low, high = (drop, high) if area > keep * full else (low, drop)
    return np.clip(texture - drop, 0, None) / (1 - drop)


def add_fog(image, severity, seed=None, shape=None):
    """Fogging on `image` at `severity`; `shape` is one of FOG_SHAPES (random if None).
    Returns (distorted_image, full-strength mask)."""
    if seed is not None:
        _seed_all(seed)
    keep, density, start, blur = FOG_SEVERITY[severity]
    shape = shape or str(np.random.choice(sorted(FOG_SHAPES)))
    texture = generate_texture(mod=FOG_SHAPES[shape])
    texture = np.asarray(texture[0] if isinstance(texture, tuple) else texture, dtype=np.float32)
    texture = texture.max(axis=2) if texture.ndim == 3 else texture
    H, W = image.shape[:2]
    big = cv.resize(image, (DROPLET_WORK_WIDTH, round(H * DROPLET_WORK_WIDTH / W)),
                    interpolation=cv.INTER_CUBIC).astype(np.float32)
    h, w = big.shape[:2]
    veil_shape = cv.GaussianBlur(cv.resize(texture, (w, h)), (0, 0), 0.04 * w)
    veil_shape = (veil_shape - veil_shape.min()) / (veil_shape.max() - veil_shape.min() + 1e-6)
    if keep < 1:
        veil_shape = _shrink_veil(veil_shape, keep)
    a = np.clip((veil_shape - start) / (0.6 - start), 0, 1)
    mask = a * a * (3 - 2 * a)  # soft edges
    alpha = (mask * density)[:, :, np.newaxis]
    soft = cv.GaussianBlur(big, (0, 0), blur)
    light = cv.GaussianBlur(big, (0, 0), 0.08 * w)
    light = light + (255 - light) * FOG_BRIGHTEN
    out = big * (1 - alpha) + (soft * 0.5 + light * 0.5) * alpha
    shrink = (W, H)
    delta = cv.resize(out, shrink, interpolation=cv.INTER_AREA) - cv.resize(big, shrink, interpolation=cv.INTER_AREA)
    result = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return result, cv.resize(mask.astype(np.float32), shrink, interpolation=cv.INTER_AREA)


# The approved cleaning-streaks class (our own design; neither Zhang et al. nor its repo has one).
# A badly wiped pane: wipe strokes follow arcs around a pivot outside the image; inside a stroke a
# thin film of residue with finer lines along the wipe scatters light, so the scene is softened and
# turns milky there (the overall blur SIDL reports for fingerprints). Severity = number of strokes.
STREAK_SEVERITY = {"low": (1, 1.40), "medium": (3, 1.60), "high": (6, 1.80)}  # wipe strokes, density


def add_cleaning_streaks(image, severity, seed=None):
    """Cleaning streaks on `image` at `severity`; the strokes of a lighter severity are also
    in the heavier ones. Returns (distorted_image, full-strength mask)."""
    count, density = STREAK_SEVERITY[severity]
    rng = np.random.default_rng(seed)
    H, W = image.shape[:2]
    big = cv.resize(image, (DROPLET_WORK_WIDTH, round(H * DROPLET_WORK_WIDTH / W)),
                    interpolation=cv.INTER_CUBIC).astype(np.float32)
    h, w = big.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    reach = w * rng.uniform(0.6, 1.4)  # distance of the pivot: clearly curved wipes
    cx, cy = w / 2 + reach * np.cos(angle), h / 2 + reach * np.sin(angle)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xs - cx, ys - cy)
    theta = np.arctan2(ys - cy, xs - cx) - np.arctan2(h / 2 - cy, w / 2 - cx)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi  # measured from the image centre, no wrap-around
    n_r = int(r.max()) + 2
    # Lines of residue across the radius, the same for every stroke (one cloth).
    fine = cv.GaussianBlur(rng.normal(0, 1, (1, n_r)).astype(np.float32), (0, 0), 2.5)[0]
    fine = np.clip((fine - fine.mean()) / (fine.std() + 1e-6) * 0.35 + 0.5, 0, 1) ** 1.2
    lines = 0.7 + 0.3 * fine
    broad = cv.GaussianBlur(rng.uniform(0.6, 1, (1, n_r)).astype(np.float32), (0, 0), 25)[0]
    t_min, t_max = float(theta.min()), float(theta.max())
    strokes = []
    for _ in range(STREAK_SEVERITY["high"][0]):
        width = w * rng.uniform(0.14, 0.28)
        px, py = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h  # every stroke crosses the image
        rp, tp = r[int(py), int(px)], theta[int(py), int(px)]
        r0 = rp - width * rng.uniform(0.3, 0.7)
        span = (t_max - t_min) * rng.uniform(0.5, 0.9)
        t0 = tp - span * rng.uniform(0.3, 0.7)
        strokes.append((r0, width, t0, t0 + span, rng.uniform(0.8, 1.0)))
    cover = np.zeros((h, w), np.float32)
    for r0, width, t0, t1, strength in strokes[:count]:
        across = np.clip(np.minimum(r - r0, r0 + width - r) / (0.25 * width), 0, 1)
        along = np.clip(np.minimum(theta - t0, t1 - theta) / (0.15 * (t1 - t0)), 0, 1)
        np.maximum(cover, across * along * strength, out=cover)
    ri = np.clip(r.astype(np.int32), 0, n_r - 1)
    patchy = cv.resize(rng.uniform(0, 1, (6, 10)).astype(np.float32), (w, h), interpolation=cv.INTER_CUBIC)
    patchy = np.clip(0.45 + 0.55 * patchy, 0, 1)  # the residue thins out and comes back along a wipe
    alpha = np.clip(cv.GaussianBlur(cover * lines[ri] * broad[ri] * patchy * density, (0, 0), 1.2), 0, 0.95)
    soft = cv.GaussianBlur(big, (0, 0), 8)
    haze = soft + (255 - soft) * 0.7
    out = big * (1 - alpha[:, :, None]) + (soft * 0.5 + haze * 0.5) * alpha[:, :, None]
    shrink = (W, H)
    delta = cv.resize(out, shrink, interpolation=cv.INTER_AREA) - cv.resize(big, shrink, interpolation=cv.INTER_AREA)
    result = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return result, cv.resize(alpha / 0.95, shrink, interpolation=cv.INTER_AREA).astype(np.float32)


# The approved scratch class: scratches as the capture protocol makes them, with sandpaper.
# Sanding patches (many short, roughly parallel scratches from one stroke) plus a few long single
# scratches, drawn with this project's scratch-line drawer (`_draw_tapered_streak`), with a faint
# haze where fine scratches scatter light. Severity = (sanding patches, long scratches); sanding
# only adds damage, so the scratches of a lighter severity are also in the heavier ones.
SCRATCH_SEVERITY = {"low": (1, 1), "medium": (2, 2), "high": (4, 4)}
SCRATCH_HAZE = 0.35


def generate_sanding_mask(shape, severity, seed=None):
    """[0, 1] scratch mask at `severity` (see SCRATCH_SEVERITY)."""
    if seed is not None:
        _seed_all(seed)
    patches, longs = SCRATCH_SEVERITY[severity]
    most_patches, most_longs = SCRATCH_SEVERITY["high"]
    H, W = shape
    diag = float(np.hypot(H, W))
    sanding = [(np.random.uniform(0.15, 0.85) * W, np.random.uniform(0.15, 0.85) * H, np.random.uniform(0, np.pi),
                np.random.uniform(0.12, 0.25) * diag, np.random.randint(15, 30), np.random.randint(0, 2**31 - 1))
               for _ in range(most_patches)]
    single = [(np.random.uniform(0, W), np.random.uniform(0, H), np.random.uniform(0, 2 * np.pi),
               np.random.uniform(0.2, 0.5) * diag, np.random.randint(0, 2**31 - 1))
              for _ in range(most_longs)]
    mask = np.zeros((H, W), np.float32)
    for cx, cy, angle, size, lines, sub_seed in sanding[:patches]:
        np.random.seed(sub_seed)
        for _ in range(lines):  # short scratches along the sanding direction, spread over the patch
            along, across = np.random.uniform(-0.5, 0.5) * size, np.random.uniform(-0.25, 0.25) * size
            origin = np.array([cx + along * np.cos(angle) - across * np.sin(angle),
                               cy + along * np.sin(angle) + across * np.cos(angle)])
            length = np.random.uniform(0.03, 0.10) * diag
            direction = angle + np.random.normal(0, 0.06)
            width, brightness = np.random.uniform(1, 2.2), np.random.uniform(0.5, 1.0)
            curve = _smooth_curve(n_waypoints=np.random.randint(3, 6), length=length, angle=direction,
                                  jitter=length * 0.02)
            _draw_tapered_streak(mask, curve, origin, width, brightness)
    for cx, cy, angle, length, sub_seed in single[:longs]:
        np.random.seed(sub_seed)
        width, brightness = np.random.uniform(1.5, 4), np.random.uniform(0.7, 1.0)
        curve = _smooth_curve(n_waypoints=np.random.randint(3, 6), length=length, angle=angle, jitter=length * 0.02)
        _draw_tapered_streak(mask, curve, np.array([cx, cy]), width, brightness)
    return np.clip(cv.GaussianBlur(mask, (0, 0), 0.6), 0, 1)


def add_sanding_scratches(image, severity, seed=None):
    """Scratches on `image` at `severity`, as bright light-catching lines (screen blend, as in
    `add_scratch`) with a faint haze around them. Returns (distorted_image, mask)."""
    mask = generate_sanding_mask(image.shape[:2], severity, seed=seed)
    alpha3 = cv.merge([mask, mask, mask])
    img_f = image.astype(np.float32) / 255.0
    haze = cv.GaussianBlur(mask, (0, 0), 6)[:, :, np.newaxis] * SCRATCH_HAZE
    img_f = img_f + (1 - img_f) * haze
    screen = 1 - (1 - img_f) * (1 - alpha3)
    out = img_f * (1 - alpha3 * 0.97) + screen * (alpha3 * 0.97)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8), mask


# The approved dust class: dust in lens of Zhang et al. 2025, Fig. 1(e). Speck shapes are the vendored
# `many_dust_drop` texture; drawn like the mud stains (add_mudByTxture's mask and blur steps) but lighter,
# because dust catches ambient light, and more transparent, as the paper describes; capped at 255.
# Severity (ours): layers of specks, how much each speck hides, and a faint veil from the finest dust.
DUST_SEVERITY = {"low": (1, 0.55, 0.02), "medium": (3, 0.80, 0.10), "high": (6, 0.95, 0.22)}  # layers, opacity, veil
DUST_LIGHTEN = 0.7  # how far a speck is lifted toward white


def add_dust(image, severity, seed=None):
    """Dust on `image` at `severity`; the specks of a lighter severity are also in the heavier
    ones. Returns (distorted_image, full-strength mask of the specks)."""
    if seed is not None:
        _seed_all(seed)
    layers, opacity, veil_share = DUST_SEVERITY[severity]
    H, W = image.shape[:2]
    big = cv.resize(image, (DROPLET_WORK_WIDTH, round(H * DROPLET_WORK_WIDTH / W)),
                    interpolation=cv.INTER_CUBIC).astype(np.float32)
    h, w = big.shape[:2]
    specks = np.zeros((h, w), np.float32)
    for _ in range(layers):  # drawn in order from the seed, so the first layers never change
        texture = generate_texture(mod="many_dust_drop")
        texture = np.asarray(texture[0] if isinstance(texture, tuple) else texture, dtype=np.float32)
        texture = texture.max(axis=2) if texture.ndim == 3 else texture
        layer = cv.resize((texture - texture.min()) / (texture.max() - texture.min() + 1e-6), (w, h))
        layer[layer > 0.5] = 1  # as add_mudByTxture
        np.maximum(specks, layer, out=specks)
    mask = cv.GaussianBlur(specks, (0, 0), 6)
    alpha = (mask * opacity)[:, :, np.newaxis]
    blurred = cv.GaussianBlur(big, (0, 0), 15)
    lit = blurred + (255 - blurred) * DUST_LIGHTEN
    veil = cv.GaussianBlur(big, (0, 0), 0.08 * w)
    veil = veil + (255 - veil) * 0.5
    hazed = big * (1 - veil_share) + veil * veil_share
    out = np.minimum(255, hazed * (1 - alpha) + lit * alpha)
    shrink = (W, H)
    delta = cv.resize(out, shrink, interpolation=cv.INTER_AREA) - cv.resize(big, shrink, interpolation=cv.INTER_AREA)
    result = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return result, np.clip(cv.resize(mask, shrink, interpolation=cv.INTER_AREA), 0, 1)


def render_effect(name, image, seed, severity):
    """One effect at `severity`: (distorted_image, full-strength mask). Water and mud
    (still called "dirt" in the class list) set severity through how much they cover;
    the other effects fade toward the clean image."""
    if name == "water":
        return add_water_droplets(image, severity, seed=seed)
    if name == "dirt":
        return add_mud_stains(image, severity, seed=seed)
    if name == "fogging":
        return add_fog(image, severity, seed=seed)
    if name == "cleaning_streaks":
        return add_cleaning_streaks(image, severity, seed=seed)
    if name == "scratch":
        return add_sanding_scratches(image, severity, seed=seed)
    if name == "dust":
        return add_dust(image, severity, seed=seed)
    raise ValueError(f"unknown effect {name!r}")


# physical_lens_soiling's thin-water mask is `texture + 0.2`: never below 0.2,
# anywhere in the image. The rendering uses it as the blend weight (a faint 20%
# film everywhere), but as ground truth the floor made every pixel 20% "water",
# so the same tile/cell cutoff meant far less real water than for thick water
# or dirt (Session 26 label audit). The floor is removed from the mask only.
THIN_WATER_MASK_FLOOR = 0.2


def remove_thin_water_floor(mask, floor=THIN_WATER_MASK_FLOOR):
    """Rescales a thin-water mask so its floor maps to 0 and 1 stays 1."""
    return np.clip((np.asarray(mask, dtype=np.float32) - floor) / (1.0 - floor), 0.0, 1.0)


# A labeled distortion must visibly change the image: at least MIN_VISIBLE_PIXELS
# pixels inside its mask (> 0.5) must differ from the image before the effect by
# more than MIN_PIXEL_CHANGE gray levels. Catches effects that render nothing --
# a scratch mostly outside the frame, or a bright scratch highlight on a white
# sky -- while every faint-but-visible distortion still counts.
MIN_VISIBLE_PIXELS = 20
MIN_PIXEL_CHANGE = 10


def visible_pixel_count(before, after, mask):
    """Pixels inside `mask` (> 0.5) that the effect changed by more than
    MIN_PIXEL_CHANGE gray levels (largest change over the color channels)."""
    region = np.asarray(mask) > 0.5
    if not region.any():
        return 0
    change = np.abs(after.astype(np.int16) - before.astype(np.int16))
    if change.ndim == 3:
        change = change.max(axis=2)
    return int((change[region] > MIN_PIXEL_CHANGE).sum())


def is_visible(before, after, mask):
    return visible_pixel_count(before, after, mask) >= MIN_VISIBLE_PIXELS
