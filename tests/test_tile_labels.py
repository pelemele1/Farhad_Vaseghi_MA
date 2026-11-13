import numpy as np

from src.soiling.tile_labels import rasterize_tile_label


def test_all_zero_mask_gives_all_zero_grid():
    mask = np.zeros((64, 64), dtype=np.float32)
    grid = rasterize_tile_label(mask, grid_h=8, grid_w=8, threshold=0.1)
    assert grid.shape == (8, 8)
    assert grid.dtype == np.uint8
    assert grid.sum() == 0


def test_full_coverage_mask_gives_all_ones_grid():
    mask = np.ones((64, 64), dtype=np.float32)
    grid = rasterize_tile_label(mask, grid_h=8, grid_w=8, threshold=0.1)
    assert (grid == 1).all()


def test_single_tile_covered_gives_single_positive_tile():
    # 64x64 mask split into an 8x8 grid -> each tile is 8x8 pixels. Fully
    # light up just the top-left tile (rows/cols 0-7).
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[0:8, 0:8] = 1.0
    grid = rasterize_tile_label(mask, grid_h=8, grid_w=8, threshold=0.5)
    assert grid[0, 0] == 1
    assert grid.sum() == 1


def test_threshold_controls_sensitivity():
    # A tile that's exactly 20% covered: passes a low threshold, fails a
    # high one.
    mask = np.zeros((80, 80), dtype=np.float32)
    tile = mask[0:8, 0:8]
    tile[0:8, 0:2] = 1.0  # 2 of 8 columns lit -> 25% coverage in that tile
    low = rasterize_tile_label(mask, grid_h=10, grid_w=10, threshold=0.1)
    high = rasterize_tile_label(mask, grid_h=10, grid_w=10, threshold=0.9)
    assert low[0, 0] == 1
    assert high[0, 0] == 0


def test_different_thresholds_give_different_results_on_thin_scratch_like_mask():
    # A thin one-pixel-wide line through an otherwise empty grid -- low
    # per-tile coverage, models why scratch needs a much lower threshold
    # than dirt/water in the Stage B dataset builder.
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[32, :] = 1.0  # one full-width row lit, 1/8 = 12.5% of any tile it crosses
    grid_dirt_style = rasterize_tile_label(mask, grid_h=8, grid_w=8, threshold=0.15)
    grid_scratch_style = rasterize_tile_label(mask, grid_h=8, grid_w=8, threshold=0.03)
    assert grid_dirt_style.sum() == 0
    assert grid_scratch_style.sum() == 8  # the whole row of tiles the line crosses
