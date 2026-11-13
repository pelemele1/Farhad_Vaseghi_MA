"""
The two-camera rig through Allied Vision's VmbPy (ships with Vimba X).

Both cameras run in software-trigger mode and work in lockstep: fire both, wait
for one frame from each, repeat. The preview and the saved images come from the
same loop, so whatever is saved is always a matched pair. The camera values come
from the XML files saved in Vimba X Viewer.
"""
import queue
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.capture.session import RIG

# Beyond the exposure itself: time allowed for readout and USB transfer.
FRAME_TIMEOUT_MARGIN_S = 5.0
# Bytes per second; a USB 3 link reports 450 MB/s, a USB 2 link 50 MB/s.
MIN_LINK_SPEED = 200_000_000


@dataclass
class FrameData:
    image: np.ndarray  # RGB, or None if the frame arrived incomplete
    frame_id: int
    camera_timestamp_ns: int


@dataclass
class PairData:
    frames: dict  # camera letter -> FrameData
    host_time: datetime
    trigger_gap_ms: float  # time taken to send both triggers: upper limit of the offset

    @property
    def complete(self):
        return all(f.image is not None for f in self.frames.values())


class VimbaRig:
    def __init__(self, settings_files):
        """`settings_files`: camera letter -> XML path."""
        self.settings_files = settings_files
        self._vmb = None
        self._cameras = {}
        self._streaming = []
        self._queues = {c.letter: queue.Queue() for c in RIG}
        self._timeout = FRAME_TIMEOUT_MARGIN_S

    def open(self):
        from vmbpy import PersistType, VmbSystem

        try:
            self._vmb = VmbSystem.get_instance()
            self._vmb.__enter__()
            found = {cam.get_serial(): cam for cam in self._vmb.get_all_cameras()}
            for spec in RIG:
                if spec.serial not in found:
                    raise RuntimeError(f"Camera {spec.name} (serial {spec.serial}) was not found. "
                                       "Check its USB cable.")
                cam = found[spec.serial]
                try:
                    cam.__enter__()
                except Exception as e:
                    raise RuntimeError(f"Could not open camera {spec.name}. Is Vimba X Viewer "
                                       f"still open? Close it and retry. ({e})") from e
                self._cameras[spec.letter] = cam
                self._check_usb_speed(cam, spec)
                # The Viewer saves the files while streaming, so they carry this lock,
                # which makes features read-only. Loading fails if it is already set
                # (an earlier run ended abruptly), and the load itself sets it again.
                lock = cam.get_feature_by_name("TLParamsLocked")
                lock.set(0)
                cam.load_settings(str(self.settings_files[spec.letter]), PersistType.NoLUT)
                lock.set(0)
                self._check_pixel_format(cam, spec)
                self._arm_software_trigger(cam)
            longest = max(c.get_feature_by_name("ExposureTime").get() for c in self._cameras.values())
            self._timeout = longest / 1e6 + FRAME_TIMEOUT_MARGIN_S
            for spec in RIG:
                cam = self._cameras[spec.letter]
                cam.start_streaming(self._make_handler(spec.letter), buffer_count=4)
                self._streaming.append(cam)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check_usb_speed(cam, spec):
        speed = cam.get_feature_by_name("DeviceLinkSpeed").get()
        if speed < MIN_LINK_SPEED:
            raise RuntimeError(f"Camera {spec.name} is connected at USB 2 speed ({speed / 1e6:.0f} MB/s). "
                               "Unplug it and plug it firmly into a USB 3 port, using a USB 3 cable, "
                               "then press Reconnect cameras.")

    @staticmethod
    def _check_pixel_format(cam, spec):
        name = str(cam.get_feature_by_name("PixelFormat").get())
        if name != "RGB8":
            raise RuntimeError(f"Camera {spec.name} is set to {name}; the app expects RGB8.")

    @staticmethod
    def _arm_software_trigger(cam):
        cam.get_feature_by_name("TriggerSelector").set("FrameStart")
        cam.get_feature_by_name("TriggerMode").set("On")
        cam.get_feature_by_name("TriggerSource").set("Software")

    def _make_handler(self, letter):
        from vmbpy import FrameStatus

        def handler(cam, stream, frame):
            try:
                image = None
                if frame.get_status() == FrameStatus.Complete:
                    image = np.array(frame.as_numpy_ndarray())
                self._queues[letter].put(FrameData(image, frame.get_id(), frame.get_timestamp()))
            finally:
                cam.queue_frame(frame)
        return handler

    def _drain(self):
        for q in self._queues.values():
            while not q.empty():
                q.get_nowait()

    def grab_pair(self):
        """One matched pair. Raises queue.Empty if a camera sends nothing."""
        self._drain()
        host_time = datetime.now()
        start = time.perf_counter()
        for spec in RIG:
            self._cameras[spec.letter].get_feature_by_name("TriggerSoftware").run()
        gap_ms = (time.perf_counter() - start) * 1000
        frames = {letter: q.get(timeout=self._timeout) for letter, q in self._queues.items()}
        return PairData(frames, host_time, gap_ms)

    def settings_snapshot(self):
        """Values currently in the cameras, for the log."""
        snapshot = {}
        for spec in RIG:
            cam = self._cameras[spec.letter]
            selector = cam.get_feature_by_name("BalanceRatioSelector")
            ratio = cam.get_feature_by_name("BalanceRatio")
            selector.set("Red")
            red = ratio.get()
            selector.set("Blue")
            blue = ratio.get()
            snapshot[f"{spec.position}_exposure_us"] = round(cam.get_feature_by_name("ExposureTime").get(), 1)
            snapshot[f"{spec.position}_gain_db"] = round(cam.get_feature_by_name("Gain").get(), 3)
            snapshot[f"{spec.position}_balance_red"] = round(red, 4)
            snapshot[f"{spec.position}_balance_blue"] = round(blue, 4)
        return snapshot

    def close(self):
        """Leaves the cameras in free-run, so Vimba X Viewer works normally afterwards."""
        for cam in self._streaming:
            try:
                cam.stop_streaming()
            except Exception:
                pass
        self._streaming = []
        for cam in self._cameras.values():
            try:
                cam.get_feature_by_name("TriggerMode").set("Off")
            except Exception:
                pass  # must not stop the release below: an unreleased camera hangs the program
            cam.__exit__(None, None, None)
        self._cameras = {}
        if self._vmb is not None:
            self._vmb.__exit__(None, None, None)
            self._vmb = None
