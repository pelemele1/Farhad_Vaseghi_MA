import numpy as np

from src.soiling.effects import MUD_SHAPES, add_cleaning_streaks, add_dirt, add_dust, add_fog, add_sanding_scratches, generate_sanding_mask, add_mud_stains, add_water, add_water_droplets, render_effect


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
    seen = set()
    for seed in range(30):
        np.random.seed(seed)
        mechanism = np.random.choice(["thick", "thin", "droplet"])
        seen.add(mechanism)
    assert seen == {"thick", "thin", "droplet"}


def _scene(h=120, w=180):
    """Smooth gradients plus edges, so droplets have something to change."""
    x, y = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    image = np.stack([x, y, 255 - x], axis=2).astype(np.uint8)
    image[::12] = 0
    image[:, ::16] = 255
    return image


def _changed_share(before, after):
    return float((np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2) > 10).mean())


def test_water_droplets_cover_more_at_each_severity():
    image = _scene()
    shares = [_changed_share(image, add_water_droplets(image, s, seed=4)[0]) for s in ("low", "medium", "high")]
    assert 0 < shares[0] < shares[1] < shares[2]


def test_water_droplets_are_reproducible_and_keep_the_shape():
    image = _scene()
    out1, mask1 = add_water_droplets(image, "medium", seed=9)
    out2, mask2 = add_water_droplets(image, "medium", seed=9)
    assert np.array_equal(out1, out2) and np.array_equal(mask1, mask2)
    assert out1.shape == image.shape and mask1.shape == image.shape[:2]
    assert mask1.dtype == np.float32 and 0 <= mask1.min() and mask1.max() <= 1


def test_generator_uses_droplets_for_water():
    image = _scene()
    expected, _ = add_water_droplets(image, "low", seed=3)
    assert np.array_equal(render_effect("water", image, 3, "low")[0], expected)


def test_mud_stains_cover_more_at_each_severity():
    image = _scene()
    shares = [_changed_share(image, add_mud_stains(image, s, seed=6, shape="scattered_spots")[0])
              for s in ("low", "medium", "high")]
    assert 0 < shares[0] < shares[1] < shares[2]


def test_mud_stains_are_reproducible_and_use_our_shape_names():
    image = _scene()
    out1, mask1 = add_mud_stains(image, "medium", seed=2)
    out2, mask2 = add_mud_stains(image, "medium", seed=2)
    assert np.array_equal(out1, out2) and np.array_equal(mask1, mask2)
    assert mask1.shape == image.shape[:2] and 0 <= mask1.min() and mask1.max() <= 1
    assert set(MUD_SHAPES) == {"scattered_spots", "large_splashes", "single_stain"}


def test_generator_uses_mud_stains_for_mud():
    image = _scene()
    expected, _ = add_mud_stains(image, "low", seed=5)
    assert np.array_equal(render_effect("dirt", image, 5, "low")[0], expected)


def test_fog_covers_more_at_each_severity():
    image = _scene()
    shares = [_changed_share(image, add_fog(image, s, seed=8, shape="dense_mist")[0])
              for s in ("low", "medium", "high")]
    assert 0 < shares[0] < shares[1] < shares[2]


def test_fog_is_reproducible_and_reachable_from_the_generator():
    image = _scene()
    out1, mask1 = add_fog(image, "medium", seed=4)
    out2, mask2 = add_fog(image, "medium", seed=4)
    assert np.array_equal(out1, out2) and np.array_equal(mask1, mask2)
    assert mask1.shape == image.shape[:2] and 0 <= mask1.min() and mask1.max() <= 1
    assert np.array_equal(render_effect("fogging", image, 4, "medium")[0], out1)


def test_cleaning_streaks_cover_more_at_each_severity():
    image = _scene()
    shares = [_changed_share(image, add_cleaning_streaks(image, s, seed=12)[0]) for s in ("low", "medium", "high")]
    assert 0 < shares[0] < shares[1] < shares[2]


def test_cleaning_streaks_are_reproducible_and_reachable_from_the_generator():
    image = _scene()
    out1, mask1 = add_cleaning_streaks(image, "medium", seed=7)
    out2, mask2 = add_cleaning_streaks(image, "medium", seed=7)
    assert np.array_equal(out1, out2) and np.array_equal(mask1, mask2)
    assert mask1.shape == image.shape[:2] and 0 <= mask1.min() and mask1.max() <= 1
    assert np.array_equal(render_effect("cleaning_streaks", image, 7, "medium")[0], out1)


def test_sanding_scratches_add_damage_at_each_severity():
    image = np.full((240, 360, 3), 40, np.uint8)  # dark, so the bright scratches show
    masks = [generate_sanding_mask(image.shape[:2], s, seed=3) for s in ("low", "medium", "high")]
    assert 0 < (masks[0] > 0.5).sum() < (masks[1] > 0.5).sum() < (masks[2] > 0.5).sum()
    assert np.all(masks[2] >= masks[0] - 1e-6)  # light's scratches are also in heavy


def test_sanding_scratches_are_reproducible_and_reachable_from_the_generator():
    image = np.full((240, 360, 3), 40, np.uint8)
    out1, mask1 = add_sanding_scratches(image, "medium", seed=2)
    out2, _ = add_sanding_scratches(image, "medium", seed=2)
    assert np.array_equal(out1, out2) and mask1.shape == image.shape[:2]
    assert np.array_equal(render_effect("scratch", image, 2, "medium")[0], out1)


def test_dust_covers_more_at_each_severity():
    image = _scene()
    shares = [_changed_share(image, add_dust(image, s, seed=10)[0]) for s in ("low", "medium", "high")]
    assert 0 < shares[0] < shares[1] < shares[2]


def test_dust_is_reproducible_and_reachable_from_the_generator():
    image = _scene()
    out1, mask1 = add_dust(image, "medium", seed=6)
    out2, mask2 = add_dust(image, "medium", seed=6)
    assert np.array_equal(out1, out2) and np.array_equal(mask1, mask2)
    assert mask1.shape == image.shape[:2] and 0 <= mask1.min() and mask1.max() <= 1
    assert np.array_equal(render_effect("dust", image, 6, "medium")[0], out1)
