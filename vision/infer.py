"""
infer.py -- detect beats on an (already filtered) ECG with YOLO, or with the classical detector.

All detections are returned in SECONDS (not pixels), so any plotting library can draw them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import peaks                               # noqa: E402
from vision.labels import CLASSES                   # noqa: E402
from vision.render import render_strip              # noqa: E402


@dataclass
class Detection:
    label: str
    conf: float
    t0: float            # seconds from the start of the signal passed in
    t1: float

    @property
    def t(self) -> float:
        return 0.5 * (self.t0 + self.t1)


def load_yolo(weights: str):
    """Returns (model or None, message). Never raises: the app falls back to the classical detector."""
    p = Path(weights)
    if not p.exists():
        return None, f"weights not found: {p}"
    try:
        from ultralytics import YOLO
    except Exception as e:                           # noqa: BLE001
        return None, f"ultralytics not installed ({e.__class__.__name__}); pip install ultralytics"
    try:
        return YOLO(str(p)), f"loaded {p}"
    except Exception as e:                           # noqa: BLE001
        return None, f"could not load weights: {e}"


def yolo_detect_window(model, y_window, fs: float, conf: float = 0.25, iou: float = 0.5,
                       classes=CLASSES) -> list[Detection]:
    strip = render_strip(y_window, fs)
    res = model.predict(np.asarray(strip.image), imgsz=max(strip.size), conf=conf, iou=iou, verbose=False)[0]
    names = getattr(model, "names", None) or {i: c for i, c in enumerate(classes)}
    out = []
    for b in res.boxes:
        x0, _, x1, _ = (float(v) for v in b.xyxy[0].tolist())
        out.append(Detection(str(names[int(b.cls[0])]), float(b.conf[0]),
                             float(strip.px_to_time(x0)), float(strip.px_to_time(x1))))
    return out


def merge(dets: list[Detection], min_sep_s: float = 0.15) -> list[Detection]:
    """Keep the most confident detection among any that share (almost) the same centre time."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(abs(d.t - k.t) >= min_sep_s for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.t)


def yolo_detect_long(model, y, fs: float, window_s: float = 4.0, stride_s: float = 2.0,
                     conf: float = 0.25, iou: float = 0.5) -> list[Detection]:
    y = np.asarray(y, dtype=float)
    win, stride = int(round(window_s * fs)), int(round(stride_s * fs))
    if len(y) < win:
        raise ValueError(f"signal shorter than one {window_s:g} s window")
    dets = []
    starts = list(range(0, len(y) - win + 1, stride))
    if starts[-1] + win < len(y):
        starts.append(len(y) - win)
    for s in starts:
        for d in yolo_detect_window(model, y[s:s + win], fs, conf, iou):
            dets.append(Detection(d.label, d.conf, d.t0 + s / fs, d.t1 + s / fs))
    return merge(dets)


def classical_detect(y, fs: float, half_width_s: float = 0.20) -> list[Detection]:
    """Pan-Tompkins-style R-peak detector from the toolkit, returned as boxes labelled 'R'."""
    idx = peaks.detect_r_peaks(np.asarray(y, dtype=float), fs)
    return [Detection("R", 1.0, max(i / fs - half_width_s, 0.0), i / fs + half_width_s) for i in idx]
