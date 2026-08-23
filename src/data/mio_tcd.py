"""
Sampling a pilot subset of MIO-TCD-Localization out of the official tar
archive (github.com/... -> tcd.miovision.com/challenge/dataset.html, CC
BY-NC-SA 4.0). The archive ships as a single ~3.5GB tar with no partial-
download option, so the workflow is: download it once, then use this module
to pull out a small, reproducible, randomly-sampled subset for the Stage A
pilot dataset instead of extracting all 137,743 images.

Archive layout (`tar -tf MIO-TCD-Localization.tar`):
  MIO-TCD-Localization/train/<8-digit id>.jpg   (110,000 images, has labels)
  MIO-TCD-Localization/test/<8-digit id>.jpg    (27,743 images, no public labels)
  MIO-TCD-Localization/gt_train.csv             (ground truth, train split only)

We sample from `train/` specifically because it's the only split with public
ground truth (`gt_train.csv`) -- Stage A itself doesn't need the boxes, but
using the labeled split keeps the same pilot subset reusable for the
detection-related work later without a second download/sample pass.

gt_train.csv has no header, one row per object:
  image_id,class_name,x1,y1,x2,y2
"""
import csv
import random
import tarfile
from pathlib import Path

TRAIN_PREFIX = "MIO-TCD-Localization/train/"
GT_TRAIN_MEMBER = "MIO-TCD-Localization/gt_train.csv"


def list_train_image_ids(tar_path):
    """8-digit ids (no extension) of every image in the train/ split."""
    with tarfile.open(tar_path, "r") as tar:
        names = tar.getnames()
    ids = [
        Path(n).stem
        for n in names
        if n.startswith(TRAIN_PREFIX) and n.endswith(".jpg")
    ]
    return sorted(ids)


def sample_image_ids(all_ids, n, seed):
    """Deterministic random sample of `n` ids (or all of them if n exceeds
    the pool), for a reproducible pilot subset independent of tar iteration
    order."""
    rng = random.Random(seed)
    pool = sorted(all_ids)
    n = min(n, len(pool))
    return sorted(rng.sample(pool, n))


def filter_gt_rows(gt_rows, selected_ids):
    """Keep only ground-truth rows (image_id, class, x1, y1, x2, y2) whose
    image_id is in `selected_ids`."""
    selected = set(selected_ids)
    return [row for row in gt_rows if row[0] in selected]


def read_gt_csv(path):
    with open(path, newline="") as f:
        return [row for row in csv.reader(f) if row]


def write_gt_csv(rows, path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def extract_images(tar_path, image_ids, out_dir):
    """Extract just the given train/ image ids (not the whole archive) into
    `out_dir`, flattened (no MIO-TCD-Localization/train/ prefix)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    members_by_id = {img_id: f"{TRAIN_PREFIX}{img_id}.jpg" for img_id in image_ids}
    with tarfile.open(tar_path, "r") as tar:
        for img_id, member_name in members_by_id.items():
            member = tar.getmember(member_name)
            src = tar.extractfile(member)
            (out_dir / f"{img_id}.jpg").write_bytes(src.read())
    return sorted(out_dir.glob("*.jpg"))


def build_pilot_subset(tar_path, out_dir, n=1000, seed=0):
    """End-to-end: sample `n` train image ids, extract those images, and
    write the matching gt_train.csv rows alongside them. Returns
    (image_paths, gt_rows, selected_ids)."""
    out_dir = Path(out_dir)
    all_ids = list_train_image_ids(tar_path)
    selected_ids = sample_image_ids(all_ids, n, seed)

    images_dir = out_dir / "images"
    image_paths = extract_images(tar_path, selected_ids, images_dir)

    with tarfile.open(tar_path, "r") as tar:
        gt_member = tar.extractfile(GT_TRAIN_MEMBER)
        gt_rows = [row for row in csv.reader(gt_member.read().decode().splitlines()) if row]
    subset_gt_rows = filter_gt_rows(gt_rows, selected_ids)
    write_gt_csv(subset_gt_rows, out_dir / "gt_subset.csv")

    with open(out_dir / "manifest.txt", "w") as f:
        f.write("\n".join(selected_ids) + "\n")

    return image_paths, subset_gt_rows, selected_ids
