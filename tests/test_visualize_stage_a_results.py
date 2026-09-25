import numpy as np

from scripts.visualize_stage_a_results import (
    group_probs_by_severity,
    kind_from_scores,
    parse_training_log,
    plot_probability_by_severity,
    plot_roc_pr_curves,
    select_diverse_sample_indices,
)


def test_kind_from_scores_ground_truth_labels():
    assert kind_from_scores([1, 0, 0], ("dirt", "water", "scratch")) == ("dirt",)
    assert kind_from_scores([0, 0, 0], ("dirt", "water", "scratch")) == ("clean",)


def test_kind_from_scores_with_threshold():
    scores = [0.9, 0.2, 0.5]
    assert kind_from_scores(scores, ("dirt", "water", "scratch"), threshold=0.5) == ("dirt", "scratch")
    assert kind_from_scores(scores, ("dirt", "water", "scratch"), threshold=0.95) == ("clean",)


def test_select_diverse_sample_indices_covers_every_kind_present():
    class_names = ("dirt", "water", "scratch")
    rows = [
        {"dirt": 1, "water": 0, "scratch": 0},
        {"dirt": 0, "water": 1, "scratch": 0},
        {"dirt": 0, "water": 0, "scratch": 1},
        {"dirt": 0, "water": 0, "scratch": 0},
        {"dirt": 1, "water": 0, "scratch": 0},
        {"dirt": 0, "water": 0, "scratch": 0},
    ]
    selected = select_diverse_sample_indices(rows, class_names, per_kind=2, seed=0)

    kinds_selected = set()
    for idx in selected:
        row = rows[idx]
        active = [c for c in class_names if row[c]]
        kinds_selected.add(active[0] if active else "clean")

    assert kinds_selected == {"dirt", "water", "scratch", "clean"}
    assert len(selected) == len(set(selected))  # no duplicate indices


def test_select_diverse_sample_indices_caps_at_available_count():
    class_names = ("dirt", "water", "scratch")
    rows = [{"dirt": 0, "water": 1, "scratch": 0}]  # only one "water" row exists
    selected = select_diverse_sample_indices(rows, class_names, per_kind=5, seed=0)
    assert selected == [0]


def test_parse_training_log_extracts_epoch_lines_in_order():
    log_text = """
### Starting TaskPrologue of job 1791674 on tg084
train=3200 val=400 pos_weight=[3.0, 3.0, 3.0]
epoch 1/20  train_loss=0.6531  val_loss=0.4310  (42.0s)
epoch 2/20  train_loss=0.3725  val_loss=0.3402  (36.0s)
some warning: FutureWarning blah blah
epoch 3/20  train_loss=0.3027  val_loss=0.2884  (36.9s)
wrote checkpoints/stage_a/stage_a_head.pt
=== JOB_STATISTICS ===
"""
    records = parse_training_log(log_text)
    assert [r["epoch"] for r in records] == [1, 2, 3]
    assert records[0] == {"epoch": 1, "train_loss": 0.6531, "val_loss": 0.4310}
    assert records[2]["train_loss"] == 0.3027


def test_parse_training_log_returns_empty_list_for_no_matches():
    assert parse_training_log("nothing relevant here\njust some text") == []


def test_plot_roc_pr_curves_writes_a_file(tmp_path):
    # Smoke test only (matplotlib output isn't otherwise unit-tested in this
    # project) -- confirms it runs end to end on ordinary multi-class input
    # and actually writes an image, not that the pixels are correct.
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(50, 3))
    probs = rng.random((50, 3))
    out_path = tmp_path / "curves.jpg"

    plot_roc_pr_curves(labels, probs, ("dirt", "water", "scratch"), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_roc_pr_curves_skips_degenerate_class_without_crashing(tmp_path):
    # A class with zero positives (or zero negatives) has no defined ROC/PR
    # curve -- must be skipped, not raise.
    labels = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])  # class 1 (index 1) is all-negative
    probs = np.array([[0.8, 0.1], [0.2, 0.3], [0.7, 0.4], [0.1, 0.2]])
    out_path = tmp_path / "curves_degenerate.jpg"

    plot_roc_pr_curves(labels, probs, ("a", "b"), out_path)

    assert out_path.exists()


def test_group_probs_by_severity_groups_and_orders_by_level():
    rows = [
        {"dirt_severity": "high"},
        {"dirt_severity": "none"},
        {"dirt_severity": "low"},
        {"dirt_severity": "none"},
    ]
    probs = [0.9, 0.1, 0.5, 0.2]

    grouped = group_probs_by_severity(rows, probs, "dirt")

    assert list(grouped.keys()) == ["none", "low", "high"]  # in SEVERITY_LEVELS_WITH_NONE order
    assert grouped["none"] == [0.1, 0.2]
    assert grouped["low"] == [0.5]
    assert grouped["high"] == [0.9]


def test_group_probs_by_severity_omits_absent_levels():
    rows = [{"dirt_severity": "high"}, {"dirt_severity": "high"}]
    probs = [0.8, 0.9]

    grouped = group_probs_by_severity(rows, probs, "dirt")

    assert set(grouped.keys()) == {"high"}


def test_group_probs_by_severity_falls_back_for_rows_predating_the_column():
    # A dataset built before severity columns existed at all (real case:
    # the current data/processed/stage_b_scratch15 on disk, mid-rebuild as
    # of Session 20 Round 3) has no "dirt_severity" key whatsoever -- must
    # not crash, and should infer "high"/"none" from the plain 0/1 column.
    rows = [{"dirt": "1"}, {"dirt": "0"}, {"dirt": 1}, {"dirt": 0}]
    probs = [0.9, 0.1, 0.8, 0.2]

    grouped = group_probs_by_severity(rows, probs, "dirt")

    assert set(grouped.keys()) == {"high", "none"}
    assert grouped["high"] == [0.9, 0.8]
    assert grouped["none"] == [0.1, 0.2]


def test_plot_probability_by_severity_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"dirt_severity": lvl, "water_severity": "none"} for lvl in
            (["none"] * 5 + ["low"] * 5 + ["medium"] * 5 + ["high"] * 5)]
    probs = np.zeros((20, 2))
    probs[:, 0] = rng.random(20)
    out_path = tmp_path / "severity.jpg"

    plot_probability_by_severity(rows, probs, ("dirt", "water"), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
