import csv
import io
import tarfile

import pytest

from src.data.mio_tcd import (
    build_pilot_subset,
    extract_images,
    filter_gt_rows,
    list_train_image_ids,
    sample_image_ids,
)


def _make_fake_archive(path, train_ids, test_ids, gt_rows):
    with tarfile.open(path, "w") as tar:
        for img_id in train_ids:
            data = f"fake-jpg-bytes-{img_id}".encode()
            info = tarfile.TarInfo(name=f"MIO-TCD-Localization/train/{img_id}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for img_id in test_ids:
            data = f"fake-jpg-bytes-{img_id}".encode()
            info = tarfile.TarInfo(name=f"MIO-TCD-Localization/test/{img_id}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        buf = io.StringIO()
        csv.writer(buf).writerows(gt_rows)
        csv_bytes = buf.getvalue().encode()
        info = tarfile.TarInfo(name="MIO-TCD-Localization/gt_train.csv")
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))


@pytest.fixture
def fake_archive(tmp_path):
    train_ids = [f"{i:08d}" for i in range(10)]
    test_ids = [f"{i:08d}" for i in range(10, 13)]
    gt_rows = [
        ["00000000", "car", "1", "2", "3", "4"],
        ["00000000", "bus", "5", "6", "7", "8"],
        ["00000005", "pedestrian", "1", "1", "2", "2"],
    ]
    archive_path = tmp_path / "MIO-TCD-Localization.tar"
    _make_fake_archive(archive_path, train_ids, test_ids, gt_rows)
    return archive_path, train_ids, test_ids, gt_rows


def test_list_train_image_ids_excludes_test_split(fake_archive):
    archive_path, train_ids, test_ids, _ = fake_archive
    listed = list_train_image_ids(archive_path)
    assert listed == sorted(train_ids)
    assert not set(test_ids) & set(listed)


def test_sample_image_ids_is_deterministic_and_sized():
    all_ids = [f"{i:08d}" for i in range(100)]
    a = sample_image_ids(all_ids, n=10, seed=0)
    b = sample_image_ids(all_ids, n=10, seed=0)
    c = sample_image_ids(all_ids, n=10, seed=1)
    assert a == b
    assert len(a) == 10
    assert a != c


def test_sample_image_ids_caps_at_pool_size():
    all_ids = [f"{i:08d}" for i in range(5)]
    sampled = sample_image_ids(all_ids, n=1000, seed=0)
    assert sorted(sampled) == sorted(all_ids)


def test_filter_gt_rows_keeps_only_selected_ids():
    gt_rows = [
        ["00000000", "car", "1", "2", "3", "4"],
        ["00000001", "bus", "5", "6", "7", "8"],
    ]
    filtered = filter_gt_rows(gt_rows, ["00000001"])
    assert filtered == [["00000001", "bus", "5", "6", "7", "8"]]


def test_extract_images_writes_only_requested_ids(fake_archive, tmp_path):
    archive_path, train_ids, _, _ = fake_archive
    out_dir = tmp_path / "images"
    selected = train_ids[:3]
    paths = extract_images(archive_path, selected, out_dir)
    assert len(paths) == 3
    assert {p.stem for p in paths} == set(selected)
    for p in paths:
        assert p.read_bytes() == f"fake-jpg-bytes-{p.stem}".encode()


def test_build_pilot_subset_end_to_end(fake_archive, tmp_path):
    archive_path, train_ids, _, gt_rows = fake_archive
    out_dir = tmp_path / "pilot"
    image_paths, subset_gt_rows, selected_ids = build_pilot_subset(
        archive_path, out_dir, n=4, seed=0
    )
    assert len(image_paths) == 4
    assert len(selected_ids) == 4
    assert set(selected_ids).issubset(set(train_ids))
    assert all(row[0] in selected_ids for row in subset_gt_rows)
    assert (out_dir / "gt_subset.csv").exists()
    assert (out_dir / "manifest.txt").read_text().splitlines() == selected_ids
