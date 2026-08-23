import numpy as np

from src.soiling.effects import add_dirt, add_water


def _sample_image(h=128, w=192):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_add_dirt_changes_the_image_and_returns_valid_mask():
    img = _sample_image()
    out, mask = add_dirt(img, seed=1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert not np.array_equal(out, img)


def test_add_dirt_is_reproducible_with_seed():
    img = _sample_image()
    out1, mask1 = add_dirt(img, seed=7)
    out2, mask2 = add_dirt(img, seed=7)
    assert np.array_equal(out1, out2)
    assert np.array_equal(mask1, mask2)


def test_add_water_thick_and_thin_both_change_the_image():
    img = _sample_image()
    for mechanism in ["thick", "thin"]:
        out, mask = add_water(img, seed=2, mechanism=mechanism)
        assert out.shape == img.shape
        assert mask.shape == img.shape[:2]
        assert not np.array_equal(out, img)


def test_add_water_droplet_mechanism_changes_the_image():
    img = _sample_image()
    out, mask = add_water(img, seed=3, mechanism="droplet")
    assert out.shape == img.shape
    assert mask.shape == img.shape[:2]
    assert not np.array_equal(out, img)


def test_add_water_random_mechanism_picks_all_three_over_many_seeds():
    img = _sample_image()
    seen = set()
    for seed in range(30):
        np.random.seed(seed)
        mechanism = np.random.choice(["thick", "thin", "droplet"])
        seen.add(mechanism)
    assert seen == {"thick", "thin", "droplet"}
