"""
labels.py -- MIT-BIH beat annotations -> YOLO bounding boxes on a rendered strip.

One box per beat: centred on the annotated R-peak, +/- half_width_s wide, as tall as the trace
inside that span (plus padding). Classes follow the AAMI EC57 grouping:

    N : N L R e j        S : A a J S        V : V E        F : F        Q : / f Q

Symbols outside this table (B, r, n, ? ...) are ignored. Q is excluded by default because it is
almost empty once the paced records are removed (pass with_q=True to keep it).
"""
from __future__ import annotations

import numpy as np

from .render import Strip

AAMI_MAP = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
            "A": "S", "a": "S", "J": "S", "S": "S",
            "V": "V", "E": "V",
            "F": "F",
            "/": "Q", "f": "Q", "Q": "Q"}
CLASSES_ALL = ["N", "S", "V", "F", "Q"]
CLASSES = ["N", "S", "V", "F"]          # default (no Q)


def aami_class(symbol) -> str | None:
    return AAMI_MAP.get(str(symbol))


def beat_boxes(strip: Strip, win_start: int, beat_samples, beat_symbols, half_width_s: float = 0.20,
               classes=CLASSES, pad_px: float = 6.0, min_visible: float = 0.6):
    """
    beat_samples are ABSOLUTE sample indices; the strip covers [win_start, win_start + strip.n).
    Returns a list of (class_index, cx, cy, w, h), all normalised to 0..1 (YOLO format).
    Beats whose box would be less than `min_visible` inside the window are dropped.
    """
    w_img, h_img = strip.size
    hw = int(round(half_width_s * strip.fs))
    out = []
    for s, sym in zip(beat_samples, beat_symbols):
        cls = aami_class(sym)
        if cls is None or cls not in classes:
            continue
        c = int(s) - win_start
        if c < 0 or c >= strip.n:
            continue
        lo, hi = c - hw, c + hw
        lo_c, hi_c = max(lo, 0), min(hi, strip.n - 1)
        if (hi_c - lo_c) < min_visible * (hi - lo):
            continue
        seg = strip.ypix[lo_c:hi_c + 1]
        x0, x1 = float(strip.xpix[lo_c]), float(strip.xpix[hi_c])
        y0 = max(float(seg.min()) - pad_px, 0.0)
        y1 = min(float(seg.max()) + pad_px, h_img - 1.0)
        out.append((classes.index(cls), (x0 + x1) / 2 / w_img, (y0 + y1) / 2 / h_img,
                    (x1 - x0) / w_img, (y1 - y0) / h_img))
    return out


def to_yolo_lines(boxes) -> str:
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)


def yolo_to_pixels(box, size) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) normalised -> (x0, y0, x1, y1) pixels."""
    cx, cy, w, h = box
    W, H = size
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
