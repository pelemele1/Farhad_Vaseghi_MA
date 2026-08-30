import numpy as np

from scripts.visualize_stage_a_results import (
    kind_from_scores,
    parse_training_log,
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
