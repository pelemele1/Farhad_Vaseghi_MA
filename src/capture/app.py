"""
Capture window: live view of both cameras, the session form, and one-click
capture of a clean pair, a dirty pair and a timed series for fogging. Files and
the log are written as described in src/capture/session.py.
"""
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from xml.etree import ElementTree

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from src.capture import session as cs
from src.capture.rig import VimbaRig

MAX_LOST_IN_A_ROW = 3
LEFT_SUFFIX, RIGHT_SUFFIX = "_leftCamera_97.xml", "_rightCamera_2C.xml"
START_TIMED, STOP_TIMED = "2. Start timed capture", "Stop timed capture"
NO_SECOND_TYPE = "none"


def find_profiles(settings_dir):
    """Profile names with a settings file for both cameras (e.g. "indoor")."""
    folder = Path(settings_dir)
    names = [p.name[: -len(LEFT_SUFFIX)] for p in folder.glob(f"*{LEFT_SUFFIX}")]
    return sorted(n for n in names if (folder / f"{n}{RIGHT_SUFFIX}").is_file())


def settings_serial(path):
    """The serial number of the camera a settings file was saved from."""
    feature = ElementTree.parse(path).getroot().find(".//Feature[@Name='DeviceSerialNumber']")
    return None if feature is None else feature.get("Value")


def files_by_serial(paths):
    """Camera letter -> settings file. Matched by the serial inside each file, so a
    camera can never receive the other camera's values, whatever the files are called."""
    by_serial = {settings_serial(p): Path(p) for p in paths}
    missing = [c.name for c in cs.RIG if c.serial not in by_serial]
    if missing:
        raise ValueError(f"No settings saved from camera {', '.join(missing)} among these files.")
    return {c.letter: by_serial[c.serial] for c in cs.RIG}


def profile_files(settings_dir, profile):
    folder = Path(settings_dir)
    return files_by_serial([folder / f"{profile}{LEFT_SUFFIX}", folder / f"{profile}{RIGHT_SUFFIX}"])


def loaded_label(files):
    """Dropdown name for files picked by hand: what their names have in common."""
    prefix = os.path.commonprefix([p.stem for p in files.values()]).rstrip("_- ")
    return f"{prefix or 'custom'} (loaded)"


@dataclass
class CaptureJob:
    stage: str
    folder: Path
    count: int | None  # None: until stopped (timed series)
    prefix: str  # the conditions part of the file names; for a series, of its first pair
    prefix_at: Callable[[float], str] | None = None  # timed series: elapsed seconds -> prefix
    interval_s: float = 0
    started: datetime | None = None
    saved: int = 0

    def prefix_for(self, moment):
        """The prefix for a pair taken at `moment`, or None if it isn't to be saved. A
        timed series saves its first pair and then one per interval, named by real time."""
        if self.prefix_at is None:
            return self.prefix
        self.started = self.started or moment
        elapsed = (moment - self.started).total_seconds()
        if elapsed < self.saved * self.interval_s:
            return None
        prefix = self.prefix_at(elapsed)
        if self.saved == 0:
            self.prefix = prefix
        self.saved += 1
        return prefix


class CaptureWorker(QThread):
    """Owns the cameras. A job only starts with the next pair triggered, so a
    frame exposed before the button was pressed is never saved."""

    opened = Signal(dict)
    preview = Signal(object)
    progress = Signal(str, int)
    job_done = Signal(str, int, int)
    failed = Signal(str)

    def __init__(self, rig_factory):
        super().__init__()
        self._rig_factory = rig_factory
        self._stop = threading.Event()
        self._stop_job = threading.Event()
        self._jobs = queue.Queue()
        self.center_crop = False

    def start_job(self, job):
        self._stop_job.clear()
        self._jobs.put(job)

    def stop_job(self):
        self._stop_job.set()

    def stop(self):
        self._stop.set()

    def run(self):
        rig = None
        job = recorder = None
        lost = lost_in_a_row = 0
        try:
            rig = self._rig_factory()
            rig.open()
            self.opened.emit(rig.settings_snapshot())
            last = time.perf_counter()
            while not self._stop.is_set():
                if recorder is None and not self._jobs.empty():
                    job = self._jobs.get()
                    recorder, lost = cs.PairRecorder(job.folder, job.prefix), 0
                if recorder is not None and self._stop_job.is_set():
                    self.job_done.emit(job.stage, recorder.close(), lost)
                    job = recorder = None
                    continue
                try:
                    pair = rig.grab_pair()
                except queue.Empty:
                    pair = None
                if pair is None or not pair.complete:
                    lost += 1
                    lost_in_a_row += 1
                    if lost_in_a_row >= MAX_LOST_IN_A_ROW:
                        raise RuntimeError("The cameras stopped delivering complete frames.")
                    continue
                lost_in_a_row = 0
                now = time.perf_counter()
                self._send_preview(pair, 1 / max(now - last, 1e-6), lost)
                last = now
                prefix = job.prefix_for(pair.host_time) if recorder is not None else None
                if prefix is not None:
                    recorder.prefix = prefix
                    recorder.add(pair)
                    self.progress.emit(job.stage, recorder.count)
                    if job.count is not None and recorder.count >= job.count:
                        self.job_done.emit(job.stage, recorder.close(), lost)
                        job = recorder = None
            if recorder is not None:
                self.job_done.emit(job.stage, recorder.close(), lost)
                recorder = None
        except Exception as e:
            if recorder is not None:
                try:
                    recorder.close()
                except Exception:
                    pass
            self.failed.emit(str(e))
        finally:
            if rig is not None:
                rig.close()

    def _send_preview(self, pair, rate, lost):
        views = {}
        for letter, frame in pair.frames.items():
            small = cs.preview_image(frame.image)
            high, low = cs.clipped_percent(small)
            shown = cs.preview_image(frame.image, center_crop=True) if self.center_crop else small
            views[letter] = (shown, high, low)
        self.preview.emit({"views": views, "rate": rate, "gap": pair.trigger_gap_ms, "lost": lost})


class ImageView(QLabel):
    """Shows the newest image scaled to fit, keeping its proportions."""

    def __init__(self):
        super().__init__("Waiting for the camera...")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#202020;color:#cccccc;")
        self._pixmap = None

    def show_rgb(self, rgb):
        h, w = rgb.shape[:2]
        self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self._refit()

    def _refit(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refit()
        super().resizeEvent(event)


class CaptureWindow(QMainWindow):
    def __init__(self, root, settings_dir, rig_factory=None):
        super().__init__()
        self.root = Path(root)
        self.settings_dir = Path(settings_dir)
        self._rig_factory = rig_factory
        self.worker = None
        self.snapshot = {}
        self.settings_text = {}
        self.loaded_files = {}
        self.pending_log = []
        self.current_job = None
        self.first_pair = 1
        self.busy = False
        self.connected = False
        self.setWindowTitle("Stereo capture")
        self.resize(1500, 820)
        self._build()
        self.session.setValue(cs.next_session_no(self.root))
        self._session_changed()
        self._connect_cameras()

    # ---- layout

    def _build(self):
        self.views, self.titles, self.readouts = {}, {}, {}
        preview = QHBoxLayout()
        for spec in cs.RIG:
            column = QVBoxLayout()
            title, view, readout = QLabel(), ImageView(), QLabel("-")
            title.setAlignment(Qt.AlignCenter)
            readout.setAlignment(Qt.AlignCenter)
            column.addWidget(title)
            column.addWidget(view, 1)
            column.addWidget(readout)
            preview.addLayout(column, 1)
            self.views[spec.letter], self.titles[spec.letter], self.readouts[spec.letter] = view, title, readout
        self.instruction = QLabel()
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("font-size:15px;padding:6px;")
        self.link_status = QLabel("Connecting...")
        self.center_crop = QCheckBox("Show centre at full resolution (focus check)")
        self.center_crop.toggled.connect(self._crop_toggled)
        below = QHBoxLayout()
        below.addWidget(self.link_status, 1)
        below.addWidget(self.center_crop)
        left = QVBoxLayout()
        left.addLayout(preview, 1)
        left.addWidget(self.instruction)
        left.addLayout(below)

        self.profile = QComboBox()
        self.profile.addItems(find_profiles(self.settings_dir))
        self.profile.currentTextChanged.connect(self._connect_cameras)
        self.load_button = QPushButton("Load...")
        self.load_button.setToolTip("Choose two settings files, one saved from each camera")
        self.load_button.clicked.connect(self._load_settings_files)
        profile_row = QHBoxLayout()
        profile_row.addWidget(self.profile, 1)
        profile_row.addWidget(self.load_button)
        self.session = QSpinBox()
        self.session.setRange(1, 9999)
        self.session.valueChanged.connect(self._session_changed)
        self.ctype = QComboBox()
        self.ctype.addItems(cs.CONTAMINATION_TYPES)
        self.ctype.currentTextChanged.connect(self._refresh)
        self.severity = QComboBox()
        self.severity.addItems(cs.SEVERITIES)
        self.second_type = QComboBox()
        self.second_type.addItems([NO_SECOND_TYPE, *(t for t in cs.CONTAMINATION_TYPES if t != cs.FOGGING)])
        self.second_type.setToolTip("For a pane with two contaminations. If one of them is fogging, choose it as Type.")
        self.second_type.currentTextChanged.connect(self._refresh)
        self.second_severity = QComboBox()
        self.second_severity.addItems(cs.SEVERITIES)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.5, 600)
        self.interval.setSingleStep(1)
        self.interval.setSuffix(" s")
        self.interval.setValue(5.0)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.1, 100)
        self.distance.setSingleStep(0.5)
        self.distance.setSuffix(" cm")
        self.distance.setValue(3.0)
        self.distance.valueChanged.connect(self._refresh)
        self.material = QComboBox()
        self.material.setEditable(True)
        self.material.addItems(["glass", "polycarbonate"])
        self.repetition = QSpinBox()
        self.repetition.setRange(1, 99)
        self.pane_id = QLineEdit()
        self.pane_id.textChanged.connect(self._refresh)
        self.day_night = QComboBox()
        self.day_night.addItems(cs.LIGHTING_MODES)
        self.remarks = QLineEdit()
        self.remarks.setPlaceholderText("weather, lighting, errors")
        form = QFormLayout()
        for label, widget in [("Camera settings", profile_row), ("Session no.", self.session),
                              ("Type", self.ctype), ("Severity", self.severity),
                              ("Second type", self.second_type), ("Second severity", self.second_severity),
                              ("Fog interval", self.interval),
                              ("Pane distance", self.distance), ("Material", self.material),
                              ("Repetition", self.repetition), ("Pane ID", self.pane_id),
                              ("Day / Night", self.day_night), ("Remarks", self.remarks)]:
            form.addRow(label, widget)
        session_box = QGroupBox("Session")
        session_box.setLayout(form)

        self.clean_button = QPushButton("1. Take clean pair")
        self.dirty_button = QPushButton("2. Take dirty pair")
        self.timed_button = QPushButton(START_TIMED)
        self.next_button = QPushButton("Next session")
        self.reconnect_button = QPushButton("Reconnect cameras")
        for button in (self.clean_button, self.dirty_button, self.timed_button, self.next_button):
            button.setMinimumHeight(38)
        self.clean_button.clicked.connect(lambda: self._capture("clean"))
        self.dirty_button.clicked.connect(lambda: self._capture("dirty"))
        self.timed_button.clicked.connect(self._timed_clicked)
        self.next_button.clicked.connect(self._next_session)
        self.reconnect_button.clicked.connect(self._connect_cameras)
        self.counts = QLabel()
        steps = QVBoxLayout()
        for widget in (self.clean_button, self.dirty_button, self.timed_button, self.counts, self.next_button):
            steps.addWidget(widget)
        steps_box = QGroupBox("Capture")
        steps_box.setLayout(steps)

        side = QVBoxLayout()
        side.addWidget(session_box)
        side.addWidget(steps_box)
        side.addStretch(1)
        side.addWidget(self.reconnect_button)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(340)
        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(side_widget)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.form_widgets = [self.profile, self.load_button, self.session, self.ctype, self.second_type, self.distance,
                             self.material, self.repetition, self.pane_id, self.day_night, self.remarks,
                             self.next_button, self.reconnect_button]

    # ---- cameras

    def _connect_cameras(self):
        self._stop_worker()
        self.connected = False
        profile = self.profile.currentText()
        if not profile:
            self.link_status.setText("Not connected")
            self._show_error(f"No settings files found in {self.settings_dir}.")
            return
        files = self.loaded_files.get(profile)
        factory = self._rig_factory or (lambda: VimbaRig(files or profile_files(self.settings_dir, profile)))
        self.link_status.setText("Connecting...")
        self.worker = CaptureWorker(factory)
        self.worker.center_crop = self.center_crop.isChecked()
        self.worker.opened.connect(self._opened)
        self.worker.preview.connect(self._show_preview)
        self.worker.progress.connect(self._job_progress)
        self.worker.job_done.connect(self._job_done)
        self.worker.failed.connect(self._worker_failed)
        self.worker.start()
        self._refresh()

    def _load_settings_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose one settings file for each camera", str(self.settings_dir), "Camera settings (*.xml)")
        if not paths:
            return
        try:
            if len(paths) != 2:
                raise ValueError("Please choose exactly two files, one saved from each camera.")
            files = files_by_serial(paths)
        except (ValueError, OSError, ElementTree.ParseError) as e:
            self._show_error(f"Could not use these settings files. {e}")
            return
        label = loaded_label(files)
        self.loaded_files[label] = files
        if self.profile.findText(label) < 0:
            self.profile.addItem(label)
        if self.profile.currentText() == label:
            self._connect_cameras()
        else:
            self.profile.setCurrentText(label)

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def _opened(self, snapshot):
        self.snapshot = snapshot
        self.connected = True
        for spec in cs.RIG:
            exposure = snapshot[f"{spec.position}_exposure_us"] / 1000
            self.settings_text[spec.letter] = f"{exposure:.1f} ms, {snapshot[f'{spec.position}_gain_db']:.2f} dB"
        self._refresh()

    def _worker_failed(self, message):
        self.connected = False
        self.busy = False
        self._set_form_enabled(True)
        self.timed_button.setText(START_TIMED)
        self.link_status.setText("Not connected")
        self._refresh()
        self._show_error(message)

    def _crop_toggled(self, checked):
        if self.worker is not None:
            self.worker.center_crop = checked

    def _show_preview(self, data):
        for spec in cs.RIG:
            rgb, high, low = data["views"][spec.letter]
            self.views[spec.letter].show_rgb(rgb)
            warning = "  CLIPPING" if high > 1 or low > 5 else ""
            settings = self.settings_text.get(spec.letter, "")
            self.readouts[spec.letter].setText(f"{settings}  |  bright {high:.1f}%  dark {low:.1f}%{warning}")
        self.link_status.setText(f"Connected  |  {data['rate']:.2f} pairs/s  |  "
                                 f"trigger gap {data['gap']:.1f} ms  |  lost pairs {data['lost']}")

    # ---- session state

    def _current_dir(self):
        return cs.session_dir(self.root, self.session.value())

    def _session_changed(self):
        row = cs.last_capture(self.root, self.session.value())
        if row:
            self.ctype.setCurrentText(row["type"])
            self.severity.setCurrentText(row["severity"])
            self.second_type.setCurrentText(row["second_type"] or NO_SECOND_TYPE)
            self.second_severity.setCurrentText(row["second_severity"])
            if row["interval_s"]:
                self.interval.setValue(float(row["interval_s"]))
            self.distance.setValue(float(row["pane_distance_cm"]))
            self.material.setCurrentText(row["material"])
            self.repetition.setValue(int(row["repetition"]))
            self.pane_id.setText(row["pane_id"])
            self.day_night.setCurrentText(row["day_night"])
            self.remarks.setText(row["remarks"])
        self._refresh()

    def _next_session(self):
        self._write_log()
        self.session.setValue(cs.next_session_no(self.root))

    def _stage_count(self, stage):
        return cs.count_pairs(self._current_dir() / stage)

    def _refresh(self):
        for spec in cs.RIG:
            is_dirty = spec.letter == cs.DIRTY_CAMERA
            self.titles[spec.letter].setText(f"{spec.position.upper()} camera {spec.name} ({spec.letter}): "
                                             f"{'DIRTY' if is_dirty else 'clean'} pane")
            color = "#b03a2e" if is_dirty else "#1e8449"
            self.titles[spec.letter].setStyleSheet(
                f"background:{color};color:white;font-weight:bold;font-size:15px;padding:5px;")
        clean, dirty_n = (self._stage_count(s) for s in cs.STAGES)
        if not self.busy:
            self.counts.setText(f"Saved: clean {clean}, dirty {dirty_n}")
        fogging = self.ctype.currentText() == cs.FOGGING
        second = self._second_type()
        two_different = second != self.ctype.currentText()
        ready = self.connected and not self.busy
        has_pane = bool(self.pane_id.text().strip())
        recording = self.timed_button.text() == STOP_TIMED
        can_record = ready and has_pane and clean > 0 and two_different
        self.clean_button.setEnabled(ready and has_pane)
        self.dirty_button.setEnabled(can_record and not fogging)
        self.timed_button.setEnabled(recording or (can_record and fogging))
        self.severity.setEnabled(not self.busy and not fogging)
        self.second_severity.setEnabled(not self.busy and second is not None)
        self.interval.setEnabled(not self.busy and fogging)
        self.instruction.setText(self._next_step(clean, dirty_n, fogging, has_pane, two_different, recording))

    def _second_type(self):
        second = self.second_type.currentText()
        return None if second == NO_SECOND_TYPE else second

    def _next_step(self, clean, dirty_n, fogging, has_pane, two_different, recording):
        spec = cs.camera_by_letter(cs.DIRTY_CAMERA)
        target = f"the {spec.position.upper()} camera ({spec.name})"
        if recording:
            return (f"Taking one pair every {self.interval.value():g} s. "
                    "Press Stop timed capture when the pane is clear.")
        if not two_different:
            return "The second type must be different from the first. Change it or set it to none."
        if not has_pane:
            return f"Enter the pane ID. The test pane goes in front of {target}."
        if clean == 0:
            return ("Put clean panes in front of both cameras, then take the clean pair. "
                    f"The test pane goes in front of {target}.")
        if not self._has_clean_at_this_distance():
            return (f"No clean pair at {self.distance.value():g} cm yet. Put clean panes in front of both "
                    "cameras and take one before the dirty images at this distance.")
        if fogging:
            return (f"Fog the test pane in front of {target}, then press Start timed capture straight away. "
                    "Keep the area closer than 5 m clear.")
        if dirty_n == 0:
            return (f"Clean pair done. Take the test pane out of {target}, contaminate it away from the "
                    "camera, take the phone photo, put it back, then record. Keep the area closer than 5 m clear.")
        return ("Saved. For another severity or distance, change it in the form and capture again. "
                "When this session is finished, clean the pane and press Next session.")

    def _has_clean_at_this_distance(self):
        return any((self._current_dir() / "clean").glob(f"clean_{self.distance.value():g}cm_*"))

    def _capture_row(self, job, count):
        number = self.session.value()
        dirty = cs.DIRTY_CAMERA
        clean = job.stage == "clean"
        left, right = (cs.pair_file_name(job.prefix, self.first_pair, c) for c in cs.RIG)
        second = None if clean else self._second_type()
        timed = job.prefix_at is not None
        row = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "session_no": number, "dirty_camera": dirty,
            "material": self.material.currentText(), "pane_distance_cm": self.distance.value(),
            "type": "" if clean else self.ctype.currentText(),
            "severity": "" if clean or timed else self.severity.currentText(),
            "repetition": self.repetition.value(), "pane_id": self.pane_id.text().strip(),
            "day_night": self.day_night.currentText(), "remarks": self.remarks.text().strip(),
            "second_type": second or "", "second_severity": self.second_severity.currentText() if second else "",
            "interval_s": job.interval_s if timed else "",
            "stage": job.stage, "first_pair": self.first_pair, "pairs": count,
            "file_left": left, "file_right": right,
            "dirty_camera_name": cs.camera_by_letter(dirty).name,
            "settings_profile": self.profile.currentText(),
        }
        row.update(self.snapshot)
        return row

    def _write_log(self):
        """Rows that couldn't be written (log open in Excel) stay queued for the next try."""
        if not self.pending_log:
            return
        try:
            cs.append_log(self.root, self.pending_log)
            self.pending_log = []
        except OSError as e:
            self._show_error("Could not update the log. If it is open in Excel, close it. The images "
                             f"are saved, and the log entry is written after the next capture. ({e})")

    # ---- capture

    def _set_form_enabled(self, enabled):
        for widget in self.form_widgets:
            widget.setEnabled(enabled)

    def _prefix_maker(self, stage):
        """Takes the form's values now: the worker thread must not read the widgets."""
        second = self._second_type()
        return partial(cs.capture_prefix, stage, self.ctype.currentText(), self.severity.currentText(),
                       self.distance.value(), self.repetition.value(), second, self.second_severity.currentText())

    def _capture(self, stage):
        self._begin(CaptureJob(stage, self._current_dir() / stage, 1, self._prefix_maker(stage)()))

    def _timed_clicked(self):
        if self.timed_button.text() == STOP_TIMED:
            self.worker.stop_job()
            self.timed_button.setEnabled(False)
            return
        self.timed_button.setText(STOP_TIMED)
        make = self._prefix_maker("dirty")
        self._begin(CaptureJob("dirty", self._current_dir() / "dirty", None, make(elapsed_s=0),
                               prefix_at=lambda elapsed: make(elapsed_s=elapsed), interval_s=self.interval.value()))

    def _begin(self, job):
        self.current_job = job
        self.first_pair = cs.count_pairs(job.folder) + 1
        self.busy = True
        self._set_form_enabled(False)
        self.counts.setText(f"Recording {job.stage}: starting")
        self._refresh()
        self.worker.start_job(job)

    def _job_progress(self, stage, count):
        self.counts.setText(f"Recording {stage}: {count}")

    def _job_done(self, stage, count, lost):
        self.busy = False
        self.timed_button.setText(START_TIMED)
        self._set_form_enabled(True)
        if count:
            self.pending_log.append(self._capture_row(self.current_job, count))
            self._write_log()
        self._refresh()
        if lost:
            self.link_status.setText(f"{stage}: {count} pairs saved, {lost} incomplete pairs skipped")

    def _show_error(self, message):
        QMessageBox.warning(self, "Capture", message)

    def closeEvent(self, event):
        self._stop_worker()
        self._write_log()
        event.accept()


def run(root, settings_dir):
    app = QApplication.instance() or QApplication([])
    window = CaptureWindow(root, settings_dir)
    window.show()
    return app.exec()
