"""
Evaluates the gate-first pipeline (src/pipeline.py) on the visible-change
dataset (Session 27).

1. val split: tunes Stage A's per-class thresholds (best F1), the gate
   threshold (highest one keeping >= --recall-target of impaired images) and
   the per-class decision offsets of the tile and pixel maps (best mean F1;
   they undo the bias the training class weights give rare classes); writes
   them to --config (read by DistortionPipeline).
2. test split: Stage A image-level metrics; the gate; Stage B tile and
   Stage C pixel metrics (per class, confusion matrix, per severity), each
   without and with the gate; the image-level answer (impaired?, dominant
   distortion, which distortions). Everything goes to --out-json; per-image
   summaries go next to it (.npz) for the figure script.

Usage:
    python scripts/evaluate_visible.py --data data/processed/visible \
        --a checkpoints/visible_a/stage_a_head.pt --b checkpoints/visible_b/stage_b_head.pt \
        --c checkpoints/visible_c/stage_c_head.pt --device cuda \
        --config checkpoints/visible_pipeline.json --out-json results/visible_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.visible_dataset import VisibleDataset
from src.eval.class_maps import (
    decide,
    map_stats,
    metrics_by_severity,
    metrics_from_stats,
    stopped_stats,
    sum_stats,
    tune_class_offsets,
)
from src.eval.metrics import compute_metrics
from src.eval.severity import compute_metrics_by_severity
from src.eval.thresholds import threshold_for_recall, tune_per_class_thresholds
from src.pipeline import CLASS_NAMES, EFFECT_NAMES, DistortionPipeline, image_answer

LEVELS = ("low", "medium", "high")
GATE_TARGETS = (0.90, 0.95, 0.98, 0.99)
SHARE_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 0.1, 61)])
PIXEL_SUBSAMPLE = 8  # every 8th pixel in each direction for tuning the pixel offsets


def run_split(pipeline, data, split, batch_size, num_workers, mode, max_samples=None):
    """One pass over `split`. mode "tune": Stage A probabilities plus flattened
    tile probabilities and subsampled pixel probabilities with their labels (for
    the offsets). mode "test": Stage A probabilities plus per-image tile and
    pixel stats (ungated, with the pipeline's offsets) and pixel-map class counts."""
    ds = VisibleDataset(data, split, "pixel", max_samples=max_samples)
    tiles = VisibleDataset(data, split, "tile", max_samples=max_samples).tile_labels
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    a_probs, b_stats, c_stats, c_counts = [], [], [], []
    tune = {"b_probs": [], "b_gt": [], "c_probs": [], "c_gt": []}
    i = 0
    for images, gt in loader:
        out = pipeline(images, run_all=True)
        a_probs.append(out["a_probs"])
        for j in range(len(images)):
            b, c, g = out["b_probs"][j], out["c_probs"][j], gt[j].numpy()
            if mode == "tune":
                tune["b_probs"].append(b.reshape(b.shape[0], -1).T)
                tune["b_gt"].append(tiles[i + j].ravel())
                sub = c[:, ::PIXEL_SUBSAMPLE, ::PIXEL_SUBSAMPLE]
                tune["c_probs"].append(sub.reshape(sub.shape[0], -1).T.astype(np.float32))
                tune["c_gt"].append(g[::PIXEL_SUBSAMPLE, ::PIXEL_SUBSAMPLE].ravel())
            else:
                b_stats.append(map_stats(b, tiles[i + j], pipeline.tile_offsets))
                c_stats.append(map_stats(c, g, pipeline.pixel_offsets))
                c_counts.append(np.bincount(decide(c, pipeline.pixel_offsets).ravel(), minlength=len(CLASS_NAMES)))
        i += len(images)
        if i % 200 < len(images):
            print(f"  {split}: {i}/{len(ds)}", flush=True)
    tune = {k: np.concatenate(v) for k, v in tune.items() if v}
    return ds, np.concatenate(a_probs), b_stats, c_stats, np.array(c_counts), tune


def counts_answer(counts, gate_passed, min_share):
    """image_answer from pixel-map class counts (the answer only needs the shares)."""
    return image_answer(np.repeat(np.arange(len(CLASS_NAMES)), counts), gate_passed, min_share)


def tune_min_share(counts, gate_passed, labels):
    """Per class, the minimum share of the pixel map (over SHARE_GRID) with the
    best F1 for "is this distortion in the image" on val (gated images only can
    be positive)."""
    shares = counts[:, 1:] / counts.sum(axis=1, keepdims=True)
    tuned = {}
    for k, name in enumerate(EFFECT_NAMES):
        y = labels[:, k] > 0
        best = (-1.0, 0.0)
        for t in SHARE_GRID:
            p = gate_passed & (shares[:, k] >= t) & (shares[:, k] > 0)
            tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, float(t))
        tuned[name] = best[1]
    return tuned


def gt_answer(row):
    present = [n for n in EFFECT_NAMES if int(row[n])]
    return {"impaired": bool(present), "dominant": row["dominant"], "present": present}


def lowest_severity(row):
    levels = [row[f"{n}_severity"] for n in EFFECT_NAMES if int(row[n])]
    return min(levels, key=LEVELS.index) if levels else "none"


def answer_metrics(rows, answers):
    gts = [gt_answer(r) for r in rows]
    impaired = [g for g in gts if g["impaired"]]
    out = {
        "impaired_accuracy": float(np.mean([a["impaired"] == g["impaired"] for a, g in zip(answers, gts)])),
        "dominant_accuracy": float(np.mean([a["dominant"] == g["dominant"]
                                            for a, g in zip(answers, gts) if g["impaired"]])),
        "n_impaired": len(impaired), "n_clean": len(gts) - len(impaired),
        "clean_answered_clean": float(np.mean([not a["impaired"] for a, g in zip(answers, gts)
                                               if not g["impaired"]])),
        "exact_set_accuracy": float(np.mean([sorted(a["shares"]) == sorted(g["present"])
                                             for a, g in zip(answers, gts)])),
        "presence": {}, "dominant_by_severity": {},
    }
    for name in EFFECT_NAMES:
        t = np.array([name in g["present"] for g in gts])
        p = np.array([name in a["shares"] for a in answers])
        tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out["presence"][name] = {"precision": float(prec), "recall": float(rec),
                                 "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}
    for level in LEVELS:
        pairs = [(a, g) for a, g, r in zip(answers, gts, rows)
                 if g["impaired"] and r[f"{g['dominant']}_severity"] == level]
        if pairs:
            out["dominant_by_severity"][level] = float(np.mean([a["dominant"] == g["dominant"] for a, g in pairs]))
    return out


def map_results(rows, stats, impaired):
    """Overall / per-severity metrics and confusion matrix, without and with the gate."""
    gated = [s if keep else stopped_stats(s) for s, keep in zip(stats, impaired)]
    res = {}
    for name, lst in (("ungated", stats), ("gated", gated)):
        total = sum_stats(lst)
        res[name] = {"per_class": metrics_from_stats(total, CLASS_NAMES),
                     "confusion": total["conf"].tolist(),
                     "by_severity": metrics_by_severity(rows, lst, CLASS_NAMES)}
    clean_idx = [i for i, r in enumerate(rows) if r["dominant"] == "clean"]
    for name, lst in (("ungated", stats), ("gated", gated)):
        flagged = {n: float(np.mean([lst[i]["conf"][:, k].sum() > 0 for i in clean_idx]))
                   for k, n in enumerate(CLASS_NAMES) if k}
        res[name]["clean_images_flagged"] = flagged
    return res


def print_map_table(title, res):
    print(f"\n{title}")
    for gate in ("ungated", "gated"):
        print(f"[{gate}] class      precision  recall     f1    IoU     AP   support")
        for r in res[gate]["per_class"]:
            print(f"         {r['class']:<9} {r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:6.3f} "
                  f"{r['iou']:6.3f} {r['ap']:6.3f} {r['support']:9d}")
        for name, levels in res[gate]["by_severity"].items():
            print("         " + name + ": " + "  ".join(
                f"{lv} R={m['recall']:.3f} F1={m['f1']:.3f} AP={m['ap']:.3f}" for lv, m in levels.items()))
        print(f"         clean test images with any location flagged: {res[gate]['clean_images_flagged']}")
    print("confusion (gated, rows = truth, cols = prediction; clean/dirt/water/scratch):")
    for row in res["gated"]["confusion"]:
        print("   " + " ".join(f"{v:11d}" for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--recall-target", type=float, default=0.98,
                        help="Share of impaired val images the gate must keep")
    parser.add_argument("--config", default="checkpoints/visible_pipeline.json")
    parser.add_argument("--out-json", default="results/visible_eval.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: first N images per split")
    args = parser.parse_args()

    pipeline = DistortionPipeline(args.a, args.b, args.c, args.weights, args.device)

    print("val: tuning Stage A and gate thresholds and the map offsets", flush=True)
    val_ds, val_a, _, _, _, tune = run_split(pipeline, args.data, "val", args.batch_size, args.num_workers, "tune",
                                             args.max_samples)
    tile_offsets, tile_f1 = tune_class_offsets(tune["b_probs"], tune["b_gt"])
    pixel_offsets, pixel_f1 = tune_class_offsets(tune["c_probs"], tune["c_gt"])
    print(f"tile offsets {np.round(tile_offsets, 2).tolist()} (val mean F1 {tile_f1:.3f}); "
          f"pixel offsets {np.round(pixel_offsets, 2).tolist()} (val mean F1 {pixel_f1:.3f})", flush=True)
    val_labels = np.stack([val_ds.image_labels(i) for i in range(len(val_ds))])
    a_thresholds = tune_per_class_thresholds(val_labels, val_a, EFFECT_NAMES)
    val_impaired = val_labels.max(axis=1) > 0
    gate_t, val_recall, val_clean_pass = threshold_for_recall(val_impaired, val_a.max(axis=1), args.recall_target)
    config = {"gate_threshold": gate_t, "stage_a_thresholds": a_thresholds, "recall_target": args.recall_target,
              "tile_offsets": tile_offsets, "pixel_offsets": pixel_offsets,
              "val_gate_recall": val_recall, "val_clean_pass": val_clean_pass,
              "val_tile_mean_f1": tile_f1, "val_pixel_mean_f1": pixel_f1}
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config).write_text(json.dumps(config, indent=2))
    pipeline.gate_threshold = gate_t
    pipeline.tile_offsets, pipeline.pixel_offsets = tile_offsets, pixel_offsets
    print("val: pixel maps with the tuned offsets, for the answer's minimum shares", flush=True)
    _, _, _, _, val_counts, _ = run_split(pipeline, args.data, "val", args.batch_size, args.num_workers, "test",
                                          args.max_samples)
    min_share = tune_min_share(val_counts, val_a.max(axis=1) >= gate_t, val_labels)
    config["min_share"] = min_share
    Path(args.config).write_text(json.dumps(config, indent=2))
    print(f"answer minimum shares {min_share}; rewrote {args.config}", flush=True)
    print(f"gate threshold {gate_t:.4f} (val: impaired kept {val_recall:.3f}, clean passed {val_clean_pass:.3f}); "
          f"Stage A thresholds {a_thresholds}; wrote {args.config}", flush=True)

    print("test: all stages", flush=True)
    ds, a_probs, b_stats, c_stats, c_counts, _ = run_split(pipeline, args.data, "test", args.batch_size,
                                                            args.num_workers, "test", args.max_samples)
    rows = ds.rows
    labels = np.stack([ds.image_labels(i) for i in range(len(ds))])
    gate_score = a_probs.max(axis=1)
    impaired = gate_score >= gate_t
    gt_impaired = labels.max(axis=1) > 0

    results = {"config": config, "n_test": len(ds)}
    results["stage_a"] = {"per_class": compute_metrics(labels, a_probs, EFFECT_NAMES, a_thresholds),
                          "by_severity": compute_metrics_by_severity(rows, labels, a_probs, EFFECT_NAMES,
                                                                     a_thresholds)}
    results["gate"] = {
        "roc_auc": float(roc_auc_score(gt_impaired, gate_score)),
        "impaired_kept": float(impaired[gt_impaired].mean()),
        "clean_passed": float(impaired[~gt_impaired].mean()),
        "kept_by_lowest_severity": {lv: float(impaired[[lowest_severity(r) == lv for r in rows]].mean())
                                    for lv in LEVELS if any(lowest_severity(r) == lv for r in rows)},
        "n_impaired": int(gt_impaired.sum()), "n_clean": int((~gt_impaired).sum()),
        "tradeoff": [],
    }
    for target in GATE_TARGETS:  # threshold set on val, measured on test
        t, _, _ = threshold_for_recall(val_impaired, val_a.max(axis=1), target)
        kept = gate_score >= t
        results["gate"]["tradeoff"].append({
            "recall_target": target, "threshold": t, "impaired_kept": float(kept[gt_impaired].mean()),
            "clean_passed": float(kept[~gt_impaired].mean()),
            "low_severity_kept": float(kept[[lowest_severity(r) == "low" for r in rows]].mean())})
    results["stage_b"] = map_results(rows, b_stats, impaired)
    results["stage_c"] = map_results(rows, c_stats, impaired)

    answers = [counts_answer(c_counts[i], bool(impaired[i]), min_share) for i in range(len(ds))]
    results["answer"] = answer_metrics(rows, answers)
    answers_any = [counts_answer(c_counts[i], bool(impaired[i]), 1e-12) for i in range(len(ds))]
    results["answer_without_min_share"] = answer_metrics(rows, answers_any)

    print("\nStage A (image level, test)")
    for r in results["stage_a"]["per_class"]:
        print(f"  {r['class']:<8} t={r['threshold']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AP={r['ap']:.3f} AUC={r['roc_auc']:.3f} n={r['support']}")
    for name, levels in results["stage_a"]["by_severity"].items():
        print(f"  {name}: " + "  ".join(f"{lv} R={m['recall']:.3f} AP={m['ap']:.3f}" for lv, m in levels.items()))
    g = results["gate"]
    print(f"\nGate: ROC-AUC {g['roc_auc']:.3f}, impaired kept {g['impaired_kept']:.3f}, clean passed "
          f"{g['clean_passed']:.3f}, kept by lowest severity {g['kept_by_lowest_severity']}")
    for t in g["tradeoff"]:
        print(f"  recall target {t['recall_target']:.2f}: threshold {t['threshold']:.3f}, impaired kept "
              f"{t['impaired_kept']:.3f}, faint kept {t['low_severity_kept']:.3f}, clean passed {t['clean_passed']:.3f}")
    print_map_table("Stage B (32-px tiles, test)", results["stage_b"])
    print_map_table("Stage C (pixels, test)", results["stage_c"])
    a = results["answer"]
    print(f"\nImage-level answer: impaired/clean correct {a['impaired_accuracy']:.3f}, dominant distortion correct "
          f"{a['dominant_accuracy']:.3f} (of {a['n_impaired']} impaired; by severity {a['dominant_by_severity']}), "
          f"clean answered clean {a['clean_answered_clean']:.3f}, exact set {a['exact_set_accuracy']:.3f}")
    print("  presence: " + "  ".join(f"{n} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
                                     for n, m in a["presence"].items()))
    b = results["answer_without_min_share"]
    print(f"  (any pixel counts: impaired/clean correct {b['impaired_accuracy']:.3f}, dominant {b['dominant_accuracy']:.3f}, "
          f"clean answered clean {b['clean_answered_clean']:.3f}, exact set {b['exact_set_accuracy']:.3f})")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    np.savez_compressed(out.with_suffix(".npz"), a_probs=a_probs, gate_score=gate_score, impaired=impaired,
                        c_counts=c_counts, paths=np.array([r["path"] for r in rows]))
    print(f"\nwrote {out} and {out.with_suffix('.npz')}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
