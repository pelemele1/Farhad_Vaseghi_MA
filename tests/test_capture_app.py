"""The capture app's bookkeeping and its window, with a fake camera rig."""
import csv
import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2 as cv
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.capture import session as cs
from src.capture.app import CaptureWindow, find_profiles, profile_files
from src.capture.rig import FrameData, PairData


class FakeRig:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def open(self):
        pass

    def grab_pair(self):
        self.calls += 1
        frames = {}
        for offset, letter in enumerate("AB"):
            image = np.full((60, 80, 3), (10 + offset, 20 + self.calls % 100, 30), np.uint8)
            frames[letter] = FrameData(image, self.calls, self.calls * 1000)
        return PairData(frames, datetime.now(), 0.5)

    def settings_snapshot(self):
        return {f"{p}_{k}": 1.0 for p in ("left", "right")
                for k in ("exposure_us", "gain_db", "balance_red", "balance_blue")}

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def pump(condition, timeout_ms=8000):
    """Runs the Qt event loop until `condition()` holds."""
    loop = QEventLoop()
    timer = QTimer(interval=10)
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    assert condition(), "timed out"


@pytest.fixture
def window(qapp, tmp_path):
    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "indoor_leftCamera_97.xml").write_text("x")
    (settings / "indoor_rightCamera_2C.xml").write_text("x")
    rigs = []

    def factory():
        rigs.append(FakeRig())
        return rigs[-1]
    win = CaptureWindow(tmp_path / "captures", settings, rig_factory=factory)
    win.rigs = rigs
    pump(lambda: win.connected)
    yield win
    win.close()


def _capture_clean(window):
    window.clean_button.click()
    pump(lambda: not window.busy and window._stage_count("clean") == 1)


def _row(number, **changes):
    row = {field: "" for field in cs.LOG_FIELDS}
    row.update(session_no=number, type="mud", **changes)
    return row


def test_only_the_right_camera_is_ever_dirty():
    spec = cs.camera_by_letter(cs.DIRTY_CAMERA)
    assert (spec.position, spec.name) == ("right", "2C")


def test_rig_names_match_the_labelled_cameras():
    assert cs.camera_by_letter("A").label == "left_97" and cs.camera_by_letter("A").serial == "05NRR"
    assert cs.camera_by_letter("B").label == "right_2C" and cs.camera_by_letter("B").serial == "05HYK"


def test_log_keeps_every_capture_in_order(tmp_path):
    cs.append_log(tmp_path, [_row(2, severity="light")])
    cs.append_log(tmp_path, [_row(2, severity="heavy"), _row(3)])
    assert [r["severity"] for r in cs.read_log(tmp_path)] == ["light", "heavy", ""]
    assert cs.last_capture(tmp_path, 2)["severity"] == "heavy" and cs.last_capture(tmp_path, 9) is None


def test_file_names_carry_the_conditions():
    assert cs.capture_prefix("dirty", "water_drops", "medium", 3.0, 1) == "water_drops_medium_3cm_r1"
    assert cs.capture_prefix("dirty", "fogging", "heavy", 2.5, 2, elapsed_s=12.4) == "fogging_t012s_2.5cm_r2"
    assert (cs.capture_prefix("dirty", "water_drops", "light", 3.0, 1, "mud", "heavy")
            == "water_drops_light_and_mud_heavy_3cm_r1")
    assert (cs.capture_prefix("dirty", "fogging", "heavy", 3.0, 1, "mud", "light", elapsed_s=5)
            == "fogging_t005s_and_mud_light_3cm_r1")
    assert cs.capture_prefix("clean", "mud", "heavy", 6.0, 1, "dust", "light") == "clean_6cm_r1"
    assert cs.pair_file_name("clean_6cm_r1", 12, cs.RIG[1]) == "clean_6cm_r1_0012_right_2C.png"


def test_next_session_counts_folders_and_log(tmp_path):
    assert cs.next_session_no(tmp_path) == 1
    (tmp_path / "session_0004").mkdir()
    assert cs.next_session_no(tmp_path) == 5
    cs.append_log(tmp_path, [_row(9)])
    assert cs.next_session_no(tmp_path) == 10


def test_recorder_saves_exact_pixels_in_correct_colors(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (40, 60, 3), np.uint8)
    frames = {"A": FrameData(image, 5, 500), "B": FrameData(image[::-1].copy(), 6, 600)}
    recorder = cs.PairRecorder(tmp_path / "clean", "clean_3cm_r1")
    recorder.add(PairData(frames, datetime(2026, 11, 8), 1.25))
    assert recorder.close() == 1
    saved = cv.imread(str(tmp_path / "clean" / "clean_3cm_r1_0001_left_97.png"), cv.IMREAD_COLOR)
    assert np.array_equal(cv.cvtColor(saved, cv.COLOR_BGR2RGB), image)
    with open(tmp_path / "clean" / "pairs.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["right_2C_file"] == "clean_3cm_r1_0001_right_2C.png" and row["left_97_frame_id"] == "5"
    assert cs.count_pairs(tmp_path / "clean") == 1


def test_recorder_continues_the_numbering_in_a_folder(tmp_path):
    image = np.zeros((8, 8, 3), np.uint8)
    pair = PairData({"A": FrameData(image, 1, 1), "B": FrameData(image, 1, 1)}, datetime(2026, 11, 8), 1.0)
    for prefix in ("mud_light_3cm_r1", "mud_heavy_6cm_r1"):
        recorder = cs.PairRecorder(tmp_path / "dirty", prefix)
        recorder.add(pair)
        recorder.close()
    assert (tmp_path / "dirty" / "mud_light_3cm_r1_0001_left_97.png").is_file()
    assert (tmp_path / "dirty" / "mud_heavy_6cm_r1_0002_right_2C.png").is_file()
    assert cs.count_pairs(tmp_path / "dirty") == 2


def test_clipping_and_preview():
    image = np.zeros((100, 100, 3), np.uint8)
    image[:25] = 255
    image[25:50] = 100
    high, low = cs.clipped_percent(image)
    assert high == 25 and low == 50
    big = np.zeros((3036, 4024, 3), np.uint8)
    assert cs.preview_image(big).shape[1] == 900
    assert cs.preview_image(big, center_crop=True).shape[:2] == (674, 900)


def _settings_xml(serial):
    return f'<ModuleSettings><Feature Name="DeviceSerialNumber" Value="{serial}" Type="String" /></ModuleSettings>'


def test_profiles_need_both_cameras(tmp_path):
    (tmp_path / "a_leftCamera_97.xml").write_text("x")
    (tmp_path / "a_rightCamera_2C.xml").write_text("x")
    (tmp_path / "b_leftCamera_97.xml").write_text("x")
    assert find_profiles(tmp_path) == ["a"]


def test_settings_go_to_the_camera_they_were_saved_from(tmp_path):
    # File names crossed relative to the serials inside them.
    (tmp_path / "a_leftCamera_97.xml").write_text(_settings_xml("05HYK"))
    (tmp_path / "a_rightCamera_2C.xml").write_text(_settings_xml("05NRR"))
    files = profile_files(tmp_path, "a")
    assert files["A"].name == "a_rightCamera_2C.xml" and files["B"].name == "a_leftCamera_97.xml"


def test_settings_from_an_unknown_camera_are_refused(tmp_path):
    (tmp_path / "a_leftCamera_97.xml").write_text(_settings_xml("05HYK"))
    (tmp_path / "a_rightCamera_2C.xml").write_text(_settings_xml("05HYK"))
    with pytest.raises(ValueError, match="97"):
        profile_files(tmp_path, "a")


def _settings_pair(folder):
    left = folder / "outdoor_cam1.xml"
    right = folder / "outdoor_cam2.xml"
    left.write_text(_settings_xml("05HYK"))
    right.write_text(_settings_xml("05NRR"))
    return [str(left), str(right)]


def test_load_button_adds_any_two_settings_files(window, tmp_path, monkeypatch):
    paths = _settings_pair(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (paths, ""))
    before = len(window.rigs)
    window.load_button.click()
    assert window.profile.currentText() == "outdoor_cam (loaded)"
    assert window.loaded_files["outdoor_cam (loaded)"]["A"].name == "outdoor_cam2.xml"
    pump(lambda: len(window.rigs) > before and window.connected)


@pytest.mark.parametrize("count", [1, 3])
def test_load_button_needs_exactly_two_files(window, tmp_path, monkeypatch, count):
    paths = (_settings_pair(tmp_path) * 2)[:count]
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (paths, ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: errors.append(a[2]))
    window.load_button.click()
    assert errors and window.profile.currentText() == "indoor"


def test_buttons_follow_the_protocol_order(window):
    assert not window.clean_button.isEnabled()  # needs a pane ID
    window.pane_id.setText("P1")
    assert window.clean_button.isEnabled() and not window.dirty_button.isEnabled()


def _capture_dirty(window):
    before = window._stage_count("dirty")
    window.dirty_button.click()
    pump(lambda: window._stage_count("dirty") == before + 1 and not window.busy)


def test_one_session_holds_different_severities_and_distances(window):
    window.pane_id.setText("P1")
    window.ctype.setCurrentText("water_drops")
    window.distance.setValue(3.0)
    _capture_clean(window)
    window.severity.setCurrentText("light")
    _capture_dirty(window)
    window.severity.setCurrentText("medium")
    _capture_dirty(window)
    window.distance.setValue(6.0)
    assert "No clean pair at 6 cm" in window.instruction.text()
    window.clean_button.click()
    pump(lambda: not window.busy and window._stage_count("clean") == 2)
    assert "No clean pair" not in window.instruction.text()
    window.severity.setCurrentText("heavy")
    _capture_dirty(window)
    folder = window.root / "session_0001"
    names = sorted(p.name for p in (folder / "dirty").glob("*left_97.png"))
    assert names == ["water_drops_heavy_6cm_r1_0003_left_97.png",
                     "water_drops_light_3cm_r1_0001_left_97.png",
                     "water_drops_medium_3cm_r1_0002_left_97.png"]
    assert (folder / "clean" / "clean_3cm_r1_0001_right_2C.png").is_file()
    assert (folder / "clean" / "clean_6cm_r1_0002_right_2C.png").is_file()
    rows = cs.read_log(window.root)
    assert [(r["stage"], r["severity"], r["pane_distance_cm"]) for r in rows] == [
        ("clean", "", "3.0"), ("dirty", "light", "3.0"), ("dirty", "medium", "3.0"),
        ("clean", "", "6.0"), ("dirty", "heavy", "6.0")]
    assert rows[1]["file_right"] == "water_drops_light_3cm_r1_0001_right_2C.png"
    assert (rows[1]["dirty_camera"], rows[1]["dirty_camera_name"]) == ("B", "2C")
    assert rows[1]["left_exposure_us"] == "1.0"
    window.next_button.click()
    assert window.session.value() == 2
    assert "DIRTY" in window.titles["B"].text() and "clean" in window.titles["A"].text()


def test_two_types_are_named_and_logged(window):
    window.pane_id.setText("P3")
    window.ctype.setCurrentText("water_drops")
    window.severity.setCurrentText("light")
    window.second_type.setCurrentText("mud")
    window.second_severity.setCurrentText("heavy")
    _capture_clean(window)
    _capture_dirty(window)
    names = [p.name for p in (window.root / "session_0001" / "dirty").glob("*right_2C.png")]
    assert names == ["water_drops_light_and_mud_heavy_3cm_r1_0001_right_2C.png"]
    clean_row, dirty_row = cs.read_log(window.root)
    assert (dirty_row["second_type"], dirty_row["second_severity"]) == ("mud", "heavy")
    assert (clean_row["second_type"], clean_row["second_severity"]) == ("", "")


def test_second_type_must_differ_from_the_first(window):
    window.pane_id.setText("P3")
    window.ctype.setCurrentText("mud")
    _capture_clean(window)
    window.second_type.setCurrentText("mud")
    assert not window.dirty_button.isEnabled() and "must be different" in window.instruction.text()
    window.second_type.setCurrentText("dust")
    assert window.dirty_button.isEnabled()


def test_fogging_cannot_be_the_second_type(window):
    assert window.second_type.findText("fogging") < 0


def test_fogging_takes_one_pair_per_interval_until_stopped(window):
    window.pane_id.setText("P2")
    window.ctype.setCurrentText("fogging")
    window.interval.setValue(0.5)
    _capture_clean(window)
    assert not window.dirty_button.isEnabled() and window.timed_button.isEnabled()
    assert not window.severity.isEnabled() and window.interval.isEnabled()
    window.timed_button.click()
    dirty = window.root / "session_0001" / "dirty"
    pump(lambda: cs.count_pairs(dirty) >= 3)
    window.timed_button.click()
    pump(lambda: not window.busy)
    with open(dirty / "pairs.csv", newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    times = [datetime.fromisoformat(p["host_time"]) for p in pairs]
    assert all((b - a).total_seconds() >= 0.4 for a, b in zip(times, times[1:]))  # not every frame
    assert pairs[0]["right_2C_file"] == "fogging_t000s_3cm_r1_0001_right_2C.png"
    assert pairs[2]["right_2C_file"] == "fogging_t001s_3cm_r1_0003_right_2C.png"
    row = cs.read_log(window.root)[-1]
    assert (row["stage"], row["severity"], row["interval_s"]) == ("dirty", "", "0.5")
    assert row["pairs"] == str(len(pairs)) and row["file_right"] == pairs[0]["right_2C_file"]
    assert window.timed_button.text() == "2. Start timed capture" and window.severity.isEnabled() is False


def test_stopping_before_the_first_pair_saves_nothing_to_the_log(window):
    window.pane_id.setText("P2")
    window.ctype.setCurrentText("fogging")
    _capture_clean(window)
    rows = len(cs.read_log(window.root))
    window.timed_button.click()
    window.timed_button.click()
    pump(lambda: not window.busy)
    assert len(cs.read_log(window.root)) - rows == window._stage_count("dirty")


def test_day_night_is_logged(window):
    assert [window.day_night.itemText(i) for i in range(window.day_night.count())] == ["day", "night"]
    window.pane_id.setText("P4")
    window.day_night.setCurrentText("night")
    _capture_clean(window)
    assert cs.read_log(window.root)[-1]["day_night"] == "night"


def test_log_entry_survives_a_locked_log_file(window, monkeypatch):
    window.pane_id.setText("P5")
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: errors.append(a[2]))

    def locked(root, rows):
        raise PermissionError("in use")
    original, cs.append_log = cs.append_log, locked
    try:
        _capture_clean(window)
    finally:
        cs.append_log = original
    assert errors and len(window.pending_log) == 1
    window._write_log()
    assert not window.pending_log and len(cs.read_log(window.root)) == 1


def test_closing_releases_the_cameras(window):
    rig = window.rigs[-1]
    window.close()
    assert rig.closed
