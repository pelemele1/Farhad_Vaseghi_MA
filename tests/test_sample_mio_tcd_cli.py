import csv
import io
import subprocess
import sys
import tarfile
from pathlib import Path


def _make_fake_archive(path, train_ids, gt_rows):
    with tarfile.open(path, "w") as tar:
        for img_id in train_ids:
            data = f"fake-jpg-bytes-{img_id}".encode()
            info = tarfile.TarInfo(name=f"MIO-TCD-Localization/train/{img_id}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        buf = io.StringIO()
        csv.writer(buf).writerows(gt_rows)
        csv_bytes = buf.getvalue().encode()
        info = tarfile.TarInfo(name="MIO-TCD-Localization/gt_train.csv")
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))


def test_sample_mio_tcd_cli_runs_end_to_end(tmp_path):
    train_ids = [f"{i:08d}" for i in range(10)]
    gt_rows = [
        ["00000000", "car", "1", "2", "3", "4"],
        ["00000005", "pedestrian", "1", "1", "2", "2"],
    ]
    archive_path = tmp_path / "MIO-TCD-Localization.tar"
    _make_fake_archive(archive_path, train_ids, gt_rows)

    out_dir = tmp_path / "pilot"
    result = subprocess.run(
        [sys.executable, "scripts/sample_mio_tcd.py",
         "--tar", str(archive_path), "--out", str(out_dir), "--n", "4", "--seed", "0"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Extracted 4 images" in result.stdout

    assert len(list((out_dir / "images").glob("*.jpg"))) == 4
    assert (out_dir / "gt_subset.csv").exists()
    assert (out_dir / "manifest.txt").exists()
