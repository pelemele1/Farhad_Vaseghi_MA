"""
Opens the two-camera capture window. Close Vimba X Viewer first: only one
program can control a camera at a time.

Usage:
    python scripts/capture_app.py
    python scripts/capture_app.py --settings-dir D:/MasterThesis_FAU/CameraSettings
    python scripts/capture_app.py --check result.txt    (no window: test one pair, write the outcome)

The double-click version in D:/MasterThesis_FAU/StereoCapture is this script
packaged with PyInstaller.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture.app import find_profiles, profile_files, run  # noqa: E402
from src.capture.rig import VimbaRig  # noqa: E402


def check(settings_dir, result_file):
    """Opens the cameras, grabs one pair and writes what happened to a file."""
    rig = None
    try:
        rig = VimbaRig(profile_files(settings_dir, find_profiles(settings_dir)[0]))
        rig.open()
        pair = rig.grab_pair()
        shapes = {letter: frame.image.shape if frame.image is not None else None
                  for letter, frame in pair.frames.items()}
        outcome = f"OK complete={pair.complete} gap_ms={pair.trigger_gap_ms:.2f} {shapes}"
    except Exception as e:
        outcome = f"FAILED: {e}"
    finally:
        if rig is not None:
            rig.close()
    Path(result_file).write_text(f"{outcome}\n", encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="D:/MasterThesis_FAU/Code/data/real_captures", help="where sessions are saved")
    parser.add_argument("--settings-dir", default="D:/MasterThesis_FAU/CameraSettings",
                        help="folder with <profile>_leftCamera_97.xml and <profile>_rightCamera_2C.xml")
    parser.add_argument("--check", metavar="FILE", help="test one pair without the window")
    args = parser.parse_args()
    if args.check:
        return check(args.settings_dir, args.check)
    return run(args.root, args.settings_dir)


if __name__ == "__main__":
    sys.exit(main())
