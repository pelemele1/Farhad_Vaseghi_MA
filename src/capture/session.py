"""
Bookkeeping for real-image capture, following the supervisor's protocol
(docs/data_collection/datacollection_surveilance_en.md): which camera gets the
dirty pane, where files go, the capture log, and saving image pairs.

One session can hold many captures with different severities and pane distances,
so each file name carries its own conditions and the log has one row per capture:
    capture_log.csv
    session_0007/clean/clean_3cm_r1_0001_left_97.png, ..._right_2C.png, pairs.csv
    session_0007/dirty/water_drops_medium_6cm_r1_0002_left_97.png, ...
    session_0007/dirty/water_drops_light_and_mud_heavy_6cm_r1_0003_left_97.png, ...  (two types)
    session_0007/dirty/fogging_t015s_3cm_r1_0004_left_97.png, ...  (seconds since the series began)
The number counts up within the folder, so a capture never replaces another.
"""
import csv
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

CONTAMINATION_TYPES = ("water_drops", "mud", "fogging", "cleaning_streaks", "scratches", "dust")
SEVERITIES = ("light", "medium", "heavy")
LIGHTING_MODES = ("day", "night")
STAGES = ("clean", "dirty")
FOGGING = "fogging"

LOG_FIELDS = [
    # The protocol's fields; type and severity are empty for a clean capture, and
    # severity is empty for fogging, where the time in the file name stands in for it.
    "datetime", "session_no", "dirty_camera", "material", "pane_distance_cm", "type",
    "severity", "repetition", "pane_id", "day_night", "remarks",
    # A second contamination type on the same pane, and the spacing of a timed series.
    "second_type", "second_severity", "interval_s",
    # What was recorded, and with which camera values.
    "stage", "first_pair", "pairs", "file_left", "file_right",
    "dirty_camera_name", "settings_profile",
    "left_exposure_us", "left_gain_db", "left_balance_red", "left_balance_blue",
    "right_exposure_us", "right_gain_db", "right_balance_red", "right_balance_blue",
]
LOG_NAME = "capture_log.csv"
_SESSION_DIR = re.compile(r"session_(\d+)$")


@dataclass(frozen=True)
class CameraSpec:
    letter: str  # protocol name, A or B
    position: str  # left or right on the rig
    name: str  # the name written on the camera
    serial: str

    @property
    def label(self):
        return f"{self.position}_{self.name}"


# A = left, B = right.
RIG = (CameraSpec("A", "left", "97", "05NRR"), CameraSpec("B", "right", "2C", "05HYK"))
# Only the right pane is ever contaminated, which replaces the protocol's pane swap.
DIRTY_CAMERA = "B"


def camera_by_letter(letter):
    return next(c for c in RIG if c.letter == letter)


def session_dir(root, session_no):
    return Path(root) / f"session_{session_no:04d}"


def read_log(root):
    """Every capture logged so far, oldest first."""
    path = Path(root) / LOG_NAME
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_capture(root, session_no):
    """The session's most recent log row, or None."""
    rows = [r for r in read_log(root) if int(r["session_no"]) == session_no]
    return rows[-1] if rows else None


def next_session_no(root):
    """One past the highest session found on disk or in the log."""
    root = Path(root)
    seen = [0]
    if root.is_dir():
        seen += [int(m.group(1)) for m in (_SESSION_DIR.match(p.name) for p in root.iterdir()) if m]
    seen += [int(r["session_no"]) for r in read_log(root)]
    return max(seen) + 1


def append_log(root, rows):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOG_NAME
    is_new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="", extrasaction="raise")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def count_pairs(stage_dir):
    """Pairs recorded in a stage folder, read from its pairs.csv."""
    path = Path(stage_dir) / "pairs.csv"
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def capture_prefix(stage, contamination, severity, distance_cm, repetition,
                   second=None, second_severity=None, elapsed_s=None):
    """The conditions part of a file name, e.g. water_drops_medium_3cm_r1. Fogging
    carries `elapsed_s`, its seconds since the series began, in place of a severity."""
    conditions = f"{distance_cm:g}cm_r{repetition}"
    if stage == "clean":
        return f"clean_{conditions}"
    first = f"t{elapsed_s:03.0f}s" if elapsed_s is not None else severity
    both = f"_and_{second}_{second_severity}" if second else ""
    return f"{contamination}_{first}{both}_{conditions}"


def pair_file_name(prefix, number, camera):
    return f"{prefix}_{number:04d}_{camera.label}.png"


class PairRecorder:
    """Saves pairs as lossless PNG on background threads. At most `max_pending`
    images wait in memory; beyond that `add` blocks, which slows the capture
    down instead of dropping pairs or running out of memory."""

    PAIR_FIELDS = ["pair", "host_time", "trigger_gap_ms"] + [
        f"{c.label}_{field}" for c in RIG for field in ("file", "frame_id", "camera_timestamp_ns")
    ]

    def __init__(self, stage_dir, prefix, workers=4, max_pending=8):
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.first = count_pairs(self.stage_dir) + 1
        csv_path = self.stage_dir / "pairs.csv"
        is_new = not csv_path.is_file()
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._slots = threading.BoundedSemaphore(max_pending)
        self._futures = []
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._file, fieldnames=self.PAIR_FIELDS)
        if is_new:
            self._csv.writeheader()
            self._file.flush()
        self.count = 0

    def _write(self, path, rgb):
        try:
            bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            if not cv.imwrite(str(path), bgr, [cv.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"could not write {path}")
        finally:
            self._slots.release()

    def add(self, pair):
        number = self.first + self.count
        self.count += 1
        row = {
            "pair": number,
            "host_time": pair.host_time.isoformat(timespec="milliseconds"),
            "trigger_gap_ms": f"{pair.trigger_gap_ms:.3f}",
        }
        for camera in RIG:
            frame = pair.frames[camera.letter]
            name = pair_file_name(self.prefix, number, camera)
            self._slots.acquire()
            self._futures.append(self._pool.submit(self._write, self.stage_dir / name, frame.image))
            row[f"{camera.label}_file"] = name
            row[f"{camera.label}_frame_id"] = frame.frame_id
            row[f"{camera.label}_camera_timestamp_ns"] = frame.camera_timestamp_ns
        self._csv.writerow(row)
        self._file.flush()

    def close(self):
        """Waits for every image; raises if any of them could not be saved."""
        self._pool.shutdown(wait=True)
        self._file.close()
        for future in self._futures:
            future.result()
        return self.count


def clipped_percent(rgb):
    """(% of pixels blown out in any channel, % fully black in all channels)."""
    return float((rgb >= 254).any(axis=2).mean() * 100), float((rgb <= 1).all(axis=2).mean() * 100)


def preview_image(rgb, width=900, center_crop=False):
    """Small image for the screen; `center_crop` shows full-resolution pixels
    from the middle of the frame, for checking focus."""
    h, w = rgb.shape[:2]
    if center_crop:
        half_w, half_h = width // 2, width * 3 // 8
        return np.array(rgb[h // 2 - half_h:h // 2 + half_h, w // 2 - half_w:w // 2 + half_w])
    return cv.resize(rgb, (width, round(h * width / w)), interpolation=cv.INTER_AREA)
