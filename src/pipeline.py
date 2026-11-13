"""
Gate-first inference pipeline (Session 27):

    image -> frozen YOLOv11-m backbone (run once, strides 2-32)
          -> Stage A (strides 8-32): P(dirt), P(water), P(scratch) visible
             gate: impaired if max P >= gate threshold, else answer "clean"
          -> if impaired:
               Stage B (strides 4-32): class per 32-px tile
               Stage C (strides 2-32): class per pixel
          -> image-level answer from the pixel map: the distortions found (each
             covering at least its minimum share of the image, tuned on val), their
             share, and the dominant one (largest visible area); "clean" if the
             gate stops the image or the map locates nothing

Class maps use one class per location: 0 clean, 1 dirt, 2 water, 3 scratch
(argmax of the 4-class softmax after per-class offsets tuned on val), matching
the visible-change ground truth.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.class_maps import decide
from src.models.backbone import build_multiscale_backbone
from src.models.distortion_head import StageADistortionHead, StageBMultiScaleHead, StageCUNetHead

CLASS_NAMES = ("clean", "dirt", "water", "scratch")
EFFECT_NAMES = CLASS_NAMES[1:]
CLASS_COLORS = {"dirt": (230, 140, 20), "water": (30, 120, 255), "scratch": (235, 20, 60)}
# Default minimum share of the image a class needs in the pixel map to count as
# found (512 x 512: ~130 pixels); the pipeline uses per-class values tuned on val.
MIN_SHARE = 0.0005


def colorize(class_map, image=None, alpha=0.6, dim=0.55):
    """RGB uint8 rendering of a class map: each distortion in its color over
    the (dimmed) image, clean left as the dimmed image (or black)."""
    class_map = np.asarray(class_map)
    base = (np.zeros(class_map.shape + (3,), np.float32) if image is None
            else image.astype(np.float32) * dim)
    for value, name in enumerate(EFFECT_NAMES, start=1):
        m = class_map == value
        if image is None:
            base[m] = CLASS_COLORS[name]
        else:
            base[m] = (1 - alpha) * image[m] + alpha * np.array(CLASS_COLORS[name], np.float32)
    return base.clip(0, 255).astype(np.uint8)


def class_shares(class_map):
    """{class: fraction of the map} for the three distortion classes."""
    counts = np.bincount(np.asarray(class_map).ravel(), minlength=len(CLASS_NAMES))
    return {name: float(counts[v]) / np.asarray(class_map).size for v, name in enumerate(CLASS_NAMES) if v}


def image_answer(class_map, gate_passed, min_share=MIN_SHARE):
    """{"impaired", "dominant", "shares"}: shares = {class: fraction of the
    image} for every distortion covering at least its `min_share` (a float, or
    a dict per class) of `class_map`; dominant = the one with the largest
    share. Clean if the gate stopped the image or nothing reaches its share."""
    if not gate_passed:
        return {"impaired": False, "dominant": "clean", "shares": {}}
    limits = min_share if isinstance(min_share, dict) else {n: min_share for n in EFFECT_NAMES}
    shares = {n: s for n, s in class_shares(class_map).items() if s >= limits[n]}
    if not shares:
        return {"impaired": False, "dominant": "clean", "shares": {}}
    return {"impaired": True, "dominant": max(shares, key=shares.get), "shares": shares}


def format_answer(answer):
    if not answer["impaired"]:
        return "clean"
    lines = ["impaired", f"dominant: {answer['dominant']}"]
    lines += [f"{n} {100 * s:.1f}%" for n, s in sorted(answer["shares"].items(), key=lambda t: -t[1])]
    return "\n".join(lines)


class DistortionPipeline(torch.nn.Module):
    """Loads the three heads around one shared frozen backbone. `config`: dict
    or JSON path with "gate_threshold", "stage_a_thresholds", "tile_offsets" and
    "pixel_offsets", "min_share" (from scripts/evaluate_visible.py, tuned on the val split)."""

    def __init__(self, a_ckpt, b_ckpt, c_ckpt, weights="weights/yolo11m.pt", device="cpu", config=None):
        super().__init__()
        ckpts = {k: torch.load(p, map_location="cpu", weights_only=False)
                 for k, p in (("a", a_ckpt), ("b", b_ckpt), ("c", c_ckpt))}
        self.taps = {k: tuple(c["taps"]) for k, c in ckpts.items()}
        self.strides = tuple(sorted(set().union(*self.taps.values())))
        self.backbone = build_multiscale_backbone(weights, self.strides)
        ch = dict(zip(self.strides, self.backbone.out_channels))

        self.head_a = StageADistortionHead([ch[s] for s in self.taps["a"]], class_names=ckpts["a"]["class_names"])
        self.head_b = StageBMultiScaleHead([ch[s] for s in self.taps["b"]], class_names=ckpts["b"]["class_names"],
                                           hidden_dim=ckpts["b"].get("hidden_dim", 32),
                                           fuse_kernel=ckpts["b"].get("fuse_kernel", 3))
        self.head_c = StageCUNetHead([ch[s] for s in self.taps["c"]], class_names=ckpts["c"]["class_names"],
                                     out_stride=self.taps["c"][0])
        for key, head in (("a", self.head_a), ("b", self.head_b), ("c", self.head_c)):
            head.load_state_dict(ckpts[key]["head_state_dict"])
        assert tuple(ckpts["b"]["class_names"]) == CLASS_NAMES == tuple(ckpts["c"]["class_names"])
        self.img_size = ckpts["c"].get("img_size", 512)
        self.eval().to(device)
        self.device = device

        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        config = config or {}
        self.gate_threshold = config.get("gate_threshold", 0.5)
        self.stage_a_thresholds = config.get("stage_a_thresholds", {n: 0.5 for n in EFFECT_NAMES})
        self.tile_offsets = config.get("tile_offsets")
        self.pixel_offsets = config.get("pixel_offsets")
        self.min_share = config.get("min_share", MIN_SHARE)

    def _select(self, feats, stage):
        return [feats[self.strides.index(s)] for s in self.taps[stage]]

    @torch.no_grad()
    def forward(self, images, run_all=False):
        """images (B, 3, S, S) in [0, 1]. Returns a dict of numpy arrays:
        a_probs (B, 3), gate_score (B,), impaired (B,) bool, and -- for the
        impaired images, or all if `run_all` -- b_probs (B, 4, g, g),
        c_probs (B, 4, S, S); rows of images the gate stops are None."""
        images = images.to(self.device)
        feats = self.backbone(images)
        a_probs = torch.sigmoid(self.head_a(self._select(feats, "a")))
        gate_score = a_probs.max(dim=1).values
        impaired = gate_score >= self.gate_threshold
        out = {"a_probs": a_probs.cpu().numpy(), "gate_score": gate_score.cpu().numpy(),
               "impaired": impaired.cpu().numpy(), "b_probs": [None] * len(images), "c_probs": [None] * len(images)}
        run = torch.ones_like(impaired) if run_all else impaired
        if run.any():
            idx = run.nonzero().flatten()
            sub = [f[idx] for f in feats]
            b = torch.softmax(self.head_b(self._select(sub, "b")), dim=1).cpu().numpy()
            c = torch.softmax(self.head_c(self._select(sub, "c")), dim=1).cpu().numpy()
            for j, i in enumerate(idx.tolist()):
                out["b_probs"][i], out["c_probs"][i] = b[j], c[j]
        return out

    def predict(self, images):
        """Per image: {"a_probs", "gate_passed", "tile_map", "pixel_map", "answer"}
        (maps all-clean when the gate stops the image)."""
        raw = self.forward(images)
        g = self.img_size // 32
        results = []
        for i in range(len(images)):
            impaired = bool(raw["impaired"][i])
            tile_map = (decide(raw["b_probs"][i], self.tile_offsets) if impaired else np.zeros((g, g), np.int64))
            pixel_map = (decide(raw["c_probs"][i], self.pixel_offsets) if impaired
                         else np.zeros((self.img_size, self.img_size), np.int64))
            results.append({"a_probs": raw["a_probs"][i], "gate_passed": impaired, "tile_map": tile_map,
                            "pixel_map": pixel_map,
                            "answer": image_answer(pixel_map, impaired, self.min_share)})
        return results
