import numpy as np

from src.soiling.effects import add_scratch, generate_scratch_mask


def _sample_image(h=128, w=192):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_mask_shape_and_range():
    mask = generate_scratch_mask((128, 192), seed=1)
    assert mask.shape == (128, 192)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_mask_is_nonempty():
    mask = generate_scratch_mask((128, 192), seed=1)
    assert mask.max() > 0.0


def test_mask_has_gaps_along_streaks():
    # a real scratch is discontinuous -- for a mask with a reasonable number
    # of scratches, not every column/row spanned by a streak's bbox should
    # be lit, i.e. the mask shouldn't be a single solid blob
    mask = generate_scratch_mask((256, 256), n_scratches=(4, 4), width_px=(2, 3), seed=2)
    lit = mask > 0.05
    assert 0.0 < lit.mean() < 0.5


def test_add_scratch_changes_the_image():
    img = _sample_image()
    out, mask = add_scratch(img, seed=3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert mask.shape == img.shape[:2]
    assert not np.array_equal(out, img)


def test_add_scratch_is_reproducible_with_seed():
    img = _sample_image()
    out1, _ = add_scratch(img, seed=42)
    out2, _ = add_scratch(img, seed=42)
    assert np.array_equal(out1, out2)


def test_zero_scratches_gives_an_empty_mask():
    # add_scratch itself no longer takes severity kwargs (matches
    # add_dirt/add_water's style: amount is randomized internally, not a
    # call-time parameter) -- exercise the zero-scratch edge case through
    # generate_scratch_mask directly instead.
    mask = generate_scratch_mask((128, 192), n_scratches=(0, 0), seed=5)
    assert mask.max() == 0.0
