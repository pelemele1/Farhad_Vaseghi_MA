"""
Screenshot of the Stereo Capture app with numbered callouts, for docs/capture_app.md.

The window runs with a simulated camera pair, so no cameras are needed: the left
camera shows a real indoor capture (data/real_captures/session_0001, if present,
otherwise a MIO-TCD photo), the right camera the same view with synthetic water
drops on its pane. A clean pair is taken first, so the window shows the state
just before the dirty capture. Writes the plain and the annotated screenshot.

Usage:
    python scripts/render_capture_app_figure.py --out-dir docs/images/capture_app
"""
import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QPoint, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.capture.app import CaptureWindow  # noqa: E402
from src.capture.rig import FrameData, PairData  # noqa: E402
from src.soiling.effects import add_water_droplets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scene():
    """(left RGB, right RGB) at about 1200 px wide."""
    real = ROOT / "data" / "real_captures" / "session_0001" / "clean" / "0001_left_97.png"
    path = real if real.is_file() else sorted((ROOT / "data" / "raw" / "mio_tcd" / "images").glob("*.jpg"))[19]
    image = cv.imread(str(path))
    image = cv.resize(image, (1200, round(image.shape[0] * 1200 / image.shape[1])), interpolation=cv.INTER_AREA)
    dirty, _ = add_water_droplets(image, "medium", seed=3)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB), cv.cvtColor(dirty, cv.COLOR_BGR2RGB)


class SimulatedRig:
    """Stands in for VimbaRig: same images every pair, about two pairs per second."""

    def __init__(self, left, right):
        self.images = {"A": left, "B": right}
        self.count = 0

    def open(self):
        pass

    def grab_pair(self):
        time.sleep(0.4)
        self.count += 1
        frames = {k: FrameData(v.copy(), self.count, self.count * 10**9) for k, v in self.images.items()}
        return PairData(frames, datetime.now(), 2.1)

    def settings_snapshot(self):  # the indoor profile's values (capture log of session 1)
        values = {"left": (6118124.2, 8.25, 2.4976, 1.6245), "right": (5884580.4, 6.781, 2.4209, 1.6753)}
        return {f"{side}_{key}": v for side, vals in values.items()
                for key, v in zip(("exposure_us", "gain_db", "balance_red", "balance_blue"), vals)}

    def close(self):
        pass


def pump(condition, timeout_ms=20000):
    loop, timer = QEventLoop(), QTimer(interval=50)
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return condition()


def box(window, widget):
    top_left = widget.mapTo(window, QPoint(0, 0))
    return top_left.x(), top_left.y(), widget.width(), widget.height()


MARGIN = 56  # white strip on each side for markers that would cover text


def annotate(image, callouts):
    """Numbered red markers on a BGR screenshot padded with a white margin on both sides.
    A callout is (x, y) inside the window, or ("left" / "right", y) for a margin marker."""
    h, w = image.shape[:2]
    out = cv.copyMakeBorder(image, 0, 0, MARGIN, MARGIN, cv.BORDER_CONSTANT, value=(255, 255, 255))
    for number, (x, y) in callouts.items():
        x = MARGIN // 2 if x == "left" else w + MARGIN + MARGIN // 2 if x == "right" else x + MARGIN
        cv.circle(out, (x, y), 16, (40, 40, 220), -1, cv.LINE_AA)
        cv.circle(out, (x, y), 16, (255, 255, 255), 2, cv.LINE_AA)
        text = str(number)
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.putText(out, text, (x - tw // 2, y + th // 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="docs/images/capture_app")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left, right = scene()
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings"
        settings.mkdir()
        (settings / "indoor_leftCamera_97.xml").write_text("x")
        (settings / "indoor_rightCamera_2C.xml").write_text("x")
        window = CaptureWindow(Path(tmp) / "captures", settings, rig_factory=lambda: SimulatedRig(left, right))
        window.resize(1500, 860)
        window.show()
        assert pump(lambda: window.connected and "pairs/s" in window.link_status.text())
        window.pane_id.setText("P1")
        window.ctype.setCurrentText("water_drops")
        window.severity.setCurrentText("medium")
        window.distance.setValue(3.0)
        window.remarks.setText("indoor test")
        window.clean_button.click()
        assert pump(lambda: not window.busy and window._stage_count("clean") == 1)
        pump(lambda: False, 1500)  # a few more previews
        shot = window.grab()
        plain = out_dir / "window.png"
        shot.save(str(plain))
        image = cv.imread(str(plain))
        scale = image.shape[1] / window.width()  # the screenshot is in device pixels (display scaling)

        def point(widget, fx=0.5, fy=0.5, dx=0, dy=0):
            x, y, w, h = box(window, widget)
            return int((x + fx * w + dx) * scale), int((y + fy * h + dy) * scale)

        def left(widget):
            return "left", point(widget)[1]

        def right(widget):
            return "right", point(widget)[1]

        callouts = {
            1: point(window.titles["A"], 0.06),
            2: point(window.views["A"], 0.08, 0.12),
            3: left(window.readouts["A"]),
            4: point(window.titles["B"], 0.06),
            5: left(window.instruction),
            6: left(window.link_status),
            7: point(window.center_crop, 0.0, 0.5, -20),
            8: right(window.profile),
            9: right(window.ctype),
            10: right(window.interval),
            11: point(window.clean_button, 0.04),
            12: point(window.timed_button, 0.04),
            13: right(window.counts),
            14: point(window.next_button, 0.04),
            15: point(window.reconnect_button, 0.04),
        }
        cv.imwrite(str(out_dir / "window_annotated.png"), annotate(image, callouts))
        window.close()
    print(f"wrote {plain} and {out_dir / 'window_annotated.png'}")


if __name__ == "__main__":
    main()
