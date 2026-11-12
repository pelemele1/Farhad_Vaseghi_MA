"""
Sample figures and measured strength of the six approved synthetic contamination
generators (src/soiling/effects.py), for docs/synthetic_generators.md.

Writes to --out-dir:
  <type>.jpg       two photos per type: clean / light / medium / heavy
  overview.jpg     one photo: clean, every type at medium, and a two-type example
and to --stats-json, for --n-stats randomly chosen photos (night scenes included):
the share of the image each type visibly changes (> MIN_PIXEL_CHANGE gray levels)
at each severity, and how often a render passes the generator's visibility check.

Usage:
    python scripts/render_generator_samples.py --source data/raw/mio_tcd/images \
        --out-dir docs/images/generators --stats-json results/generator_stats.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from src.soiling.effects import MIN_PIXEL_CHANGE, is_visible, render_effect  # noqa: E402

# Code name -> name used in the thesis and the capture protocol.
TYPES = {
    "water": "water drops",
    "dirt": "mud",
    "fogging": "fogging",
    "cleaning_streaks": "cleaning streaks",
    "scratch": "scratches",
    "dust": "dust",
}
SEVERITIES = {"low": "light", "medium": "medium", "high": "heavy"}
TILE_W, TILE_H = 480, 320


def tile(image, text):
    small = cv.resize(image, (TILE_W, TILE_H), interpolation=cv.INTER_AREA)
    cv.rectangle(small, (0, 0), (TILE_W, 28), (0, 0, 0), -1)
    cv.putText(small, text, (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv.LINE_AA)
    return small


def changed_share(before, after):
    change = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    return float((change > MIN_PIXEL_CHANGE).mean())


def pick_daylight(paths, count, rng):
    """Photos bright enough for a sample figure to show the effect, chosen reproducibly."""
    chosen = []
    for index in rng.permutation(len(paths)):
        image = cv.imread(str(paths[index]))
        if image.shape[1] >= 700 and 70 <= cv.cvtColor(image, cv.COLOR_BGR2GRAY).mean() <= 170:
            chosen.append(paths[index])
        if len(chosen) == count:
            return chosen
    return chosen


def type_figures(paths, out_dir, rng):
    photos = pick_daylight(paths, 2 * len(TYPES) + 1, rng)
    for t, (name, label) in enumerate(TYPES.items()):
        rows = []
        for k, path in enumerate(photos[2 * t:2 * t + 2]):
            image = cv.imread(str(path))
            seed = 500 + 10 * t + k
            row = [tile(image, "clean")]
            for severity, shown in SEVERITIES.items():
                row.append(tile(render_effect(name, image, seed, severity)[0], f"{label} / {shown}"))
            rows.append(np.hstack(row))
        cv.imwrite(str(out_dir / f"{name}.jpg"), np.vstack(rows), [cv.IMWRITE_JPEG_QUALITY, 88])
    image = cv.imread(str(photos[-1]))
    grid = [tile(image, "clean")]
    for t, (name, label) in enumerate(TYPES.items()):
        grid.append(tile(render_effect(name, image, 900 + t, "medium")[0], f"{label} / medium"))
    both, _ = render_effect("water", render_effect("dirt", image, 910, "medium")[0], 911, "medium")
    grid.append(tile(both, "two types: mud + water drops / medium"))
    cv.imwrite(str(out_dir / "overview.jpg"), np.vstack([np.hstack(grid[:4]), np.hstack(grid[4:])]),
               [cv.IMWRITE_JPEG_QUALITY, 88])


def measure(paths, n, rng):
    picks = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    stats = {name: {s: [] for s in SEVERITIES} for name in TYPES}
    visible = {name: {s: 0 for s in SEVERITIES} for name in TYPES}
    for p, path in enumerate(picks):
        image = cv.imread(str(path))
        for t, name in enumerate(TYPES):
            for severity in SEVERITIES:
                out, mask = render_effect(name, image, 10_000 + 100 * p + t, severity)
                stats[name][severity].append(changed_share(image, out))
                visible[name][severity] += int(is_visible(image, out, mask))
        print(f"measured photo {p + 1}/{n}", flush=True)
    summary = {}
    for name in TYPES:
        summary[name] = {}
        for severity, shown in SEVERITIES.items():
            v = np.array(stats[name][severity]) * 100
            summary[name][shown] = {"mean_pct": round(float(v.mean()), 1), "min_pct": round(float(v.min()), 1),
                                    "max_pct": round(float(v.max()), 1),
                                    "visible_first_try": f"{visible[name][severity]}/{n}"}
    return {"photos": [Path(p).name for p in picks], "changed_share": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/raw/mio_tcd/images")
    parser.add_argument("--out-dir", default="docs/images/generators")
    parser.add_argument("--stats-json", default="results/generator_stats.json")
    parser.add_argument("--n-stats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()
    np.seterr(all="ignore")  # the vendored texture code divides by zero on purpose
    paths = sorted(Path(args.source).glob("*.jpg"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    type_figures(paths, out_dir, np.random.default_rng(args.seed))
    print(f"figures written to {out_dir} ({time.time() - start:.0f}s)", flush=True)
    if args.skip_stats:
        return
    result = measure(paths, args.n_stats, np.random.default_rng(args.seed + 1))
    Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("| Type | Light | Medium | Heavy |\n|---|---|---|---|")
    for name, label in TYPES.items():
        cells = [f"{c['mean_pct']}% ({c['min_pct']}–{c['max_pct']}%)" for c in result["changed_share"][name].values()]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
